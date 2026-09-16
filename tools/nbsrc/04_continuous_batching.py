# %% [markdown]
# # 04 · Continuous Batching：从零实现迭代级调度
#
# 第 02 章证明了单个请求的 decode 根本喂不饱 GPU。这一章解决下一个问题：**怎么把多个请求拼在一起跑**。
#
# 有两种做法：
#
# | | Static Batching | Continuous Batching |
# |---|---|---|
# | 批次怎么组 | 一次性组好，整批跑完才换 | 每次迭代结束就重新组批 |
# | 短请求怎么办 | 等整批跑完，槽位空转 | 立刻释放，补进新请求 |
# | 类比 | 坐满一车人才发车的班车 | 到站就上人下人的公交车 |
#
# 这一章会把两种调度器都实现出来，用同一批请求跑，然后量化差距。

# %%
# @@SETUP@@

# %% [markdown]
# ## 一、MiniGPT 的关键能力
#
# 实现连续批处理的前提，是模型能接受**不同进度的序列混合在一个 batch 里**。这需要三样东西，引导单元里的 `MiniGPT.forward` 都已经具备：
#
# 1. `past_kvs`：每条请求带着自己的 KV cache。
# 2. `pos_offset`：支持传一个**张量**，让 batch 里每条序列用各自的绝对位置——因为它们解码的进度不一样。
# 3. `attn_mask`：支持传入自定义掩码，把补齐的填充位屏蔽掉。
#
# > 真实的 vLLM 不使用填充，而是用分页的 block table 让每条序列的 KV 物理上就不连续。填充是"能跑通"的做法，但它浪费显存——这正是第 05 章要解决的。

# %%
model = build_model(block_size=4096)
print(f"参数量 {model.n_params / 1e6:.1f}M，上下文 {model.cfg.block_size}")

# %% [markdown]
# ## 二、请求对象与单条预填充

# %%
class Req:
    """一条请求。每条请求持有自己的 KV cache，与其他请求完全独立。"""

    __slots__ = ("rid", "prompt", "max_new", "out", "past", "pos", "done")

    def __init__(self, rid, prompt, max_new):
        self.rid = rid
        self.prompt = prompt
        self.max_new = max_new
        self.out = []
        self.past = None
        self.pos = 0
        self.done = False

    def __repr__(self):
        return f"Req#{self.rid}(pos={self.pos}, out={len(self.out)}/{self.max_new}, done={self.done})"


@torch.no_grad()
def prefill_one(model, r):
    """新请求入场：整段 prompt 算一遍，产出第一个 token。"""
    logits, past = model(r.prompt)
    r.past = past
    r.pos = r.prompt.size(1)
    r.out = [logits[:, -1].argmax(-1, keepdim=True)]
    if len(r.out) >= r.max_new:
        r.done = True


def make_workload(n, prompt_len=64, lo=4, hi=65, vocab=50257, seed=0):
    """造一批输出长度差异很大的请求——差异越大，static batching 浪费越严重。"""
    g = torch.Generator().manual_seed(seed)
    reqs = []
    for i in range(n):
        prompt = torch.randint(0, vocab, (1, prompt_len), generator=g).to(DEVICE)
        max_new = int(torch.randint(lo, hi, (1,), generator=g).item())
        reqs.append(Req(i, prompt, max_new))
    return reqs


workload = make_workload(8)
print("示例工作负载的输出长度:", [r.max_new for r in workload])

# %% [markdown]
# ## 三、核心：把不同进度的请求拼成一个 batch
#
# 这是本章最难的一段代码，值得逐行读。
#
# 挑战在于：请求 A 已经解码到位置 70，请求 B 才刚入场、在位置 64。要放进同一个张量，KV 长度必须对齐，所以：
#
# 1. 把每条序列的 KV **右填充**到该 batch 的最大长度；
# 2. 构造逐序列掩码，只允许看到「自己的真实历史」和「本步新 token」，填充位全部屏蔽；
# 3. 位置偏移传成张量，每条序列用各自的真实位置；
# 4. 算完后把结果**拆回**每条请求自己的 KV，去掉填充。

# %%
@torch.no_grad()
def decode_batch(model, reqs):
    """一次 decode 迭代：批量处理一组进度不同的请求。"""
    B = len(reqs)
    lens = [r.pos for r in reqs]
    Lmax = max(lens)
    n_layer = model.cfg.n_layer

    # --- 1. 右填充到 Lmax ---
    padded = []
    for layer in range(n_layer):
        ks, vs = [], []
        for r in reqs:
            k, v = r.past[layer]
            pad = Lmax - k.size(2)
            if pad:
                k = F.pad(k, (0, 0, 0, pad))
                v = F.pad(v, (0, 0, 0, pad))
            ks.append(k)
            vs.append(v)
        padded.append((torch.cat(ks, 0), torch.cat(vs, 0)))

    # --- 2. 逐序列掩码：真实历史 [0, L_i) + 新 token 落在下标 Lmax ---
    S = Lmax + 1
    mask = torch.zeros(B, 1, 1, S, dtype=torch.bool, device=DEVICE)
    for i, r in enumerate(reqs):
        mask[i, 0, 0, : r.pos] = True
        mask[i, 0, 0, Lmax] = True

    # --- 3. 每条序列用各自的位置 ---
    nxt = torch.cat([r.out[-1] for r in reqs], 0)
    pos = torch.tensor(lens, device=DEVICE)
    logits, past = model(nxt, past_kvs=padded, pos_offset=pos, attn_mask=mask)

    # --- 4. 拆回每条请求自己的 KV，丢掉填充 ---
    for i, r in enumerate(reqs):
        rebuilt = []
        for layer in range(n_layer):
            k_all, v_all = past[layer]
            k = torch.cat([k_all[i:i + 1, :, : r.pos], k_all[i:i + 1, :, Lmax:Lmax + 1]], dim=2)
            v = torch.cat([v_all[i:i + 1, :, : r.pos], v_all[i:i + 1, :, Lmax:Lmax + 1]], dim=2)
            rebuilt.append((k, v))
        r.past = rebuilt
        r.pos += 1
        r.out.append(logits[i:i + 1, -1].argmax(-1, keepdim=True))
        if len(r.out) >= r.max_new:
            r.done = True


# 先验证它是对的：把不同长度的请求混在一起，输出必须和单条跑完全一致
cmp_reqs = make_workload(3, seed=7)
for r in cmp_reqs:
    prefill_one(model, r)

ref = [generate_cached(model, r.prompt, r.max_new) for r in cmp_reqs]

batch_reqs = make_workload(3, seed=7)
for r in batch_reqs:
    prefill_one(model, r)
while any(not r.done for r in batch_reqs):
    decode_batch(model, [r for r in batch_reqs if not r.done])

for i, (r, ref_i) in enumerate(zip(batch_reqs, ref)):
    got = torch.cat(r.out, dim=1)
    ok = torch.equal(got, ref_i[:, r.prompt.size(1):])
    print(f"Req#{i} 混批结果 == 单独跑结果 : {ok}")

# %% [markdown]
# 三条请求混批跑出来的 token，和逐条单独跑出来的**逐位一致**。这是性能优化必须做的验证——快不算本事，快且算得对才算。

# %% [markdown]
# ## 四、两种调度器

# %%
def run_static(model, reqs, batch_size):
    """Static batching：固定分组，整组跑满组内最长的输出长度。
    早完成的请求不能释放槽位，只能空转——这就是浪费的来源。"""
    slot_steps = useful_steps = 0
    for i in range(0, len(reqs), batch_size):
        group = reqs[i:i + batch_size]
        for r in group:
            prefill_one(model, r)
        longest = max(r.max_new for r in group)
        for _ in range(longest - 1):
            slot_steps += len(group)
            active = [r for r in group if not r.done]
            useful_steps += len(active)
            if active:
                decode_batch(model, active)
    return slot_steps, useful_steps


def run_continuous(model, reqs, max_batch):
    """Continuous batching：每轮迭代结束就把完成的请求踢出去，立刻补进队列里的新请求。"""
    waiting = list(reqs)
    running = []
    slot_steps = useful_steps = 0

    while waiting or running:
        while waiting and len(running) < max_batch:
            r = waiting.pop(0)
            prefill_one(model, r)
            running.append(r)

        slot_steps += len(running)
        useful_steps += sum(1 for r in running if not r.done)

        active = [r for r in running if not r.done]
        if active:
            decode_batch(model, active)

        running = [r for r in running if not r.done]

    return slot_steps, useful_steps

# %% [markdown]
# ## 五、实验：同一批请求，两种调度

# %%
N, BATCH = 32, 8

w1 = make_workload(N, seed=1)
t0 = time.perf_counter()
slot_s, useful_s = run_static(model, w1, BATCH)
t_static = time.perf_counter() - t0

w2 = make_workload(N, seed=1)
t0 = time.perf_counter()
slot_c, useful_c = run_continuous(model, w2, BATCH)
t_cont = time.perf_counter() - t0

total_tokens = sum(r.max_new for r in w1)

print(f"{N} 条请求，batch 上限 {BATCH}，输出长度 4~64 随机\n")
print(f"{'调度方式':<22}{'槽位利用率':>12}{'耗时(s)':>10}{'吞吐(t/s)':>12}")
print("-" * 58)
print(f"{'static batching':<22}{useful_s / slot_s * 100:>11.1f}%{t_static:>10.2f}{total_tokens / t_static:>12,.0f}")
print(f"{'continuous batching':<22}{useful_c / slot_c * 100:>11.1f}%{t_cont:>10.2f}{total_tokens / t_cont:>12,.0f}")
print()
print(f"槽位利用率提升: {(useful_c / slot_c) / (useful_s / slot_s):.2f}x")

# %% [markdown]
# ### 结果解读
#
# **槽位利用率一定提升**，这是 continuous batching 的机制决定的：完成的请求立刻释放，新请求立刻补位，没有空转。
#
# 但**墙钟时间不一定成比例改善**，你可能会看到耗时只提升一点点，甚至持平。这不是实现有问题，而是揭示了一个真实工程事实：
#
# 上面这个朴素的 continuous batching 实现，每轮迭代都要为整个 batch 分配 `B × Lmax` 的 KV 显存并做填充拷贝。当各序列长度差异大时，**填充本身的开销会吃掉调度带来的收益**。
#
# 这正是 vLLM 的 PagedAttention 要解决的问题——把 KV cache 切成固定大小的 block，让每条序列按需分配，物理上就不需要对齐。下一章就做这件事。

# %%
# 量化一下填充浪费：随机造几种长度分布，算 (B*Lmax) / sum(L_i)
def padding_waste(lens):
    return len(lens) * max(lens) / sum(lens)


print("同一批请求，长度分布对填充浪费的影响：")
print(f"{'长度分布':<28}{'浪费倍数':>10}")
print("-" * 40)
for label, lens in [
    ("全一样 [64]*8", [64] * 8),
    ("轻微差异 [56..64]", list(range(56, 64))),
    ("差异较大 [16..128]", [16, 32, 48, 64, 80, 96, 112, 128]),
    ("一个超长 + 7 个短", [16] * 7 + [512]),
]:
    print(f"{label:<28}{padding_waste(lens):>9.2f}x")

print()
print("最后一行是关键：只要 batch 里混进一条超长序列，整个 batch 都要按它的长度分配显存。")
print("线上流量里长尾请求始终存在，所以这不是理论问题，是每天都要付的成本。")

# %% [markdown]
# ## 六、面试话术
#
# 被问"continuous batching 相比 static batching 的收益来自哪"，这样答：
#
# 1. **机制**：static 是整批跑完才换人，短请求完成后槽位空转；continuous 每轮迭代重新组批，完成的立刻走、排队的立刻进。
#
# 2. **收益大小取决于什么**：输出长度分布的**方差**。方差越大，static 的浪费越严重。如果所有请求输出长度都一样，两者几乎没差别——这一点能答出来，说明你理解机制而不是背结论。
#
# 3. **代价**：每轮迭代都要重新组批，调度逻辑复杂度上升；batch 内序列进度不一致，KV 管理变复杂（这就是 PagedAttention 的由来）；还要处理新请求的 prefill 与老请求的 decode 争抢（这是 chunked prefill 的由来，第 06 章）。
#
# 4. **一个真实约束**：continuous 让请求延迟变得不可预测——你不知道自己的请求会和谁拼在一起，所以 p99 反而更难控制。线上必须用优先级或分池来隔离。
#
# **作业**
#
# 1. 把 `make_workload` 的输出长度范围改成 `lo=60, hi=65`（差异很小），重跑实验，看两种方式的墙钟时间差距是否消失了。
# 2. 把 `BATCH` 从 8 改成 32（等于一次性放进所有请求），continuous 会退化成什么？
# 3. 现在的新请求入场时会单独做一次 prefill，这会占掉整轮迭代的时间。想想怎么改进——这就是下一章的内容之一。
#
# **下一章**：彻底解决填充浪费，用分页的方式管理 KV cache，并顺手实现前缀复用。

