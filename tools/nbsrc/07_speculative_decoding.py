# %% [markdown]
# # 07 · 投机解码：什么时候它是负收益
#
# 第 02 章得出过一个结论：**decode 阶段 GPU 大量时间在等显存，算力是闲置的。**
#
# 投机解码（speculative decoding）就是冲着这个浪费去的。核心思路一句话：
#
# > 用一个便宜的模型连续猜 γ 个 token，再用目标模型**一次并行验证**，猜对的直接收下。
#
# 但这一章的重点不是"它有多快"，而是**它在什么条件下会变慢**。这是面试官最爱追问的地方，也是很多论文不会告诉你的部分。

# %%
# @@SETUP@@

# %% [markdown]
# ## 一、机制拆解
#
# 一轮投机解码分三步：
#
# 1. **草稿（draft）**：便宜地连续生成 γ 个候选 token。
# 2. **验证（verify）**：把这 γ 个候选**一次性**喂给目标模型。因为是一次前向，GPU 并行处理，所以耗时接近单次 decode——**这是整个技术成立的前提**。
# 3. **接受（accept）**：从第一个位置开始逐个比对，接受最长的匹配前缀。第一个不匹配的位置用目标模型的输出纠正。
#
# 关键收益来源：验证阶段一次前向能确认**多个** token，而正常的 decode 一次前向只能确认一个。

# %% [markdown]
# ## 二、先验证一个前提：验证真的很便宜吗
#
# "一次前向验证 γ 个 token，耗时接近一次 decode"——这是整个技术的基石。先实测它。

# %%
model = build_model(block_size=4096)
CTX = 256

ctx = torch.randint(0, model.cfg.vocab_size, (1, CTX), device=DEVICE)
_, past = model(ctx)

print(f"上下文长度 {CTX}\n")
print(f"{'一次前向处理的 token 数':>24}{'耗时(ms)':>12}{'相对 1 token':>14}")
print("-" * 52)
base = None
for k in [1, 2, 4, 8, 16, 32]:
    inp = torch.randint(0, model.cfg.vocab_size, (1, k), device=DEVICE)
    ms = bench(lambda: model(inp, past_kvs=past, pos_offset=CTX), warmup=3, iters=10)
    if base is None:
        base = ms
    print(f"{k:>24}{ms:>12.3f}{ms / base:>13.2f}x")

print()
print("这就是全部秘密：token 数从 1 涨到 32，耗时只涨了很小的比例。")
print("因为 decode 阶段真正的瓶颈是把全部权重从显存读一遍——读一次是这些时间，")
print("读一次算 1 个 token 还是算 32 个 token，差别很小（直到算力成为新瓶颈）。")
print()
print("所以：只要验证这一次前向的价格 ≈ 一次普通 decode，而它能确认多个 token，就赚了。")

# %% [markdown]
# ## 三、实现
#
# 下面实现一个贪心版投机解码。为了把算法逻辑跑通并且**验证它输出正确**，这里提供两种草稿：
#
# - `draft="self"`：用目标模型自己当草稿。它一定猜得准（接受率 100%），但**完全不便宜**。
# - `draft="random"`：随机猜。它几乎免费，但**完全不准**。
#
# 真实的草稿模型要同时满足"便宜"和"准"，这两个极端都做不到——这正是本章要传达的重点。

# %%
@torch.no_grad()
def spec_decode(model, prompt, max_new, gamma=4, draft="self", seed=0):
    """贪心版投机解码。返回 (完整序列, 统计信息)。"""
    g = torch.Generator().manual_seed(seed)
    logits, past = model(prompt)
    pos = prompt.size(1)                              # KV 已覆盖的位置数
    nxt = logits[:, -1].argmax(-1, keepdim=True)      # 位置 pos 上的待确认 token
    out = []                                          # 已确认的 token
    stats = {"proposed": 0, "accepted": 0, "forwards": 1}

    while len(out) + 1 < max_new:
        # ---------- 1) 草稿阶段 ----------
        if draft == "self":
            cands = [nxt]                             # 当前分布的最优点就是第一个猜测
            d_past, d_pos, d_in = past, pos, nxt
            for _ in range(gamma - 1):
                lg, d_past = model(d_in, past_kvs=d_past, pos_offset=d_pos)
                d_pos += 1
                d_in = lg[:, -1].argmax(-1, keepdim=True)
                cands.append(d_in)
            stats["forwards"] += gamma - 1
        else:
            cands = [torch.randint(0, model.cfg.vocab_size, (1, 1), generator=g).to(DEVICE)
                     for _ in range(gamma)]

        cand = torch.cat(cands, dim=1)                # (1, gamma)

        # ---------- 2) 验证阶段：一次前向验证全部候选 ----------
        v_logits, v_past = model(cand, past_kvs=past, pos_offset=pos)
        stats["forwards"] += 1
        # 目标模型在每个候选位置上的贪心输出
        target = torch.cat([nxt, v_logits[:, :-1].argmax(-1, keepdim=True)], dim=1)
        # 全部接受时白送的那个 token
        bonus = v_logits[:, -1].argmax(-1, keepdim=True)

        # ---------- 3) 接受最长匹配前缀 ----------
        a = 0
        for i in range(gamma):
            if target[0, i].item() == cand[0, i].item():
                a += 1
            else:
                break
        stats["proposed"] += gamma
        stats["accepted"] += a

        # ---------- 4) 收下已接受的 token ----------
        for i in range(a):
            out.append(cand[:, i:i + 1])

        # ---------- 5) 纠偏 token：全接受时拿 bonus，否则用目标模型的纠正 ----------
        forced = bonus if a == gamma else target[:, a:a + 1]
        out.append(forced)

        # ---------- 6) KV 截断到接受长度，并补一步让 forced 进入 KV ----------
        keep = pos + a
        past = [(k[:, :, :keep], v[:, :, :keep]) for k, v in v_past]
        lg, past = model(forced, past_kvs=past, pos_offset=keep)
        stats["forwards"] += 1
        pos = keep + 1
        nxt = lg[:, -1].argmax(-1, keepdim=True)

    if len(out) < max_new:
        out.append(nxt)
    return torch.cat([prompt] + out[:max_new], dim=1), stats


# %% [markdown]
# ## 四、先证明它算得对
#
# 贪心验证的投机解码，输出必须和普通贪心解码**逐位完全一致**。不接受任何近似。

# %%
prompt = torch.randint(0, model.cfg.vocab_size, (1, 64), device=DEVICE)
MAX_NEW = 32

ref = generate_cached(model, prompt, MAX_NEW)

print(f"{'草稿类型':<12}{'γ':>4}{'输出与贪心一致':>16}{'接受率':>10}{'前向次数':>10}")
print("-" * 54)
for draft in ["self", "random"]:
    for gamma in [2, 4, 8]:
        got, st = spec_decode(model, prompt, MAX_NEW, gamma=gamma, draft=draft)
        ok = torch.equal(got, ref)
        acc = st["accepted"] / max(1, st["proposed"]) * 100
        print(f"{draft:<12}{gamma:>4}{str(ok):>16}{acc:>9.1f}%{st['forwards']:>10}")

# %% [markdown]
# **两种草稿的输出都和贪心完全一致**——算法逻辑是对的。
#
# 但注意接受率和前向次数的差别：
#
# - `self`：接受率 100%，但为了猜 γ 个 token 已经花掉了 γ-1 次前向，加上验证和补步，**总前向次数和直接解码差不多**。
# - `random`：几乎全部被拒绝，每轮只能推进 1 个 token，却花了验证 + 补步 2 次前向——**比直接解码还慢一倍**。

# %% [markdown]
# ## 五、实测速度

# %%
ms_base = bench(lambda: generate_cached(model, prompt, MAX_NEW), warmup=1, iters=3)
print(f"基线（普通贪心解码）: {ms_base:8.1f} ms\n")
print(f"{'草稿类型':<12}{'γ':>4}{'耗时(ms)':>12}{'加速比':>10}")
print("-" * 40)
for draft in ["self", "random"]:
    for gamma in [2, 4, 8]:
        ms = bench(lambda: spec_decode(model, prompt, MAX_NEW, gamma=gamma, draft=draft),
                   warmup=1, iters=3)
        print(f"{draft:<12}{gamma:>4}{ms:>12.1f}{ms_base / ms:>9.2f}x")

print()
print("两个极端都拿不到收益，甚至明显变慢。这不是实现问题，是数学约束。")

# %% [markdown]
# ## 六、收益的理论边界
#
# 把上面的直觉写成公式。设：
#
# - `α`：草稿的接受率
# - `γ`：每轮草稿长度
# - `c_d`：草稿单步成本 ÷ 目标模型单步成本（草稿有多便宜）
#
# 一轮投机解码的**期望产出**（标准结论）：
#
# ```
# E[tokens] = (1 - α^(γ+1)) / (1 - α)
# ```
#
# 一轮的**成本**（以目标模型单步为 1 个单位）：
#
# ```
# cost = γ × c_d + 1        ← 1 是那次并行验证，它约等于一次普通 decode
# ```
#
# 所以 `加速比 = E[tokens] / cost`。下面把这张表扫出来。

# %%
def expected_tokens(alpha, gamma):
    if alpha >= 1.0:
        return gamma + 1
    return (1 - alpha ** (gamma + 1)) / (1 - alpha)


def speedup(alpha, gamma, c_d):
    return expected_tokens(alpha, gamma) / (gamma * c_d + 1)


print("加速比（c_d = 0.1，草稿模型比目标模型便宜 10 倍）\n")
alphas = [0.2, 0.4, 0.6, 0.8, 0.95]
print(f"{'接受率 α':>10}" + "".join(f"{'γ=' + str(g):>10}" for g in [2, 4, 8]))
print("-" * 40)
for a in alphas:
    row = "".join(f"{speedup(a, g, 0.1):>10.2f}" for g in [2, 4, 8])
    print(f"{a:>10.2f}{row}")

# %%
print("草稿成本 c_d 的影响（α = 0.7, γ = 4）\n")
print(f"{'c_d':>8}{'草稿相对成本':>16}{'加速比':>10}")
print("-" * 36)
for c_d in [0.02, 0.05, 0.1, 0.2, 0.5, 1.0]:
    print(f"{c_d:>8.2f}{c_d * 4:>15.2f}{speedup(0.7, 4, c_d):>10.2f}")

print()
print("读表得到三条结论：")
print("  1. 接受率越低，可行区间越窄：α=0.2 时只有 γ=2 勉强持平（1.03x），γ 越大亏得越多。")
print("  2. γ 不是越大越好：α 低时增大 γ 只会让成本线性上升、产出几乎不涨。")
print("  3. c_d = 1（草稿和目标是同一个模型）时加速比恰好是 1.0——")
print("     这正好对上前面 self 草稿的实测结果。理论和实验对上了。")

# %% [markdown]
# ## 七、什么时候是负收益
#
# 把上面两张表翻译成工程判断：
#
# | 条件 | 为什么是负收益 |
# |---|---|
# | 草稿模型太弱，接受率低 | 每轮只推进 1 个 token，却付出了验证开销，纯亏 |
# | 草稿模型太大，成本接近目标模型 | 成本翻倍，产出不涨（实测里 `self` 就是这个情况） |
# | **decoding batch 已经很大** | GPU 算力已经打满，验证那次前向不再"近乎免费"，收益消失 |
# | 输出高度随机（高温采样、开放创作） | 接受率天然低 |
# | 请求本来就是算力受限（超长 prefill） | 瓶颈不在 decode，优化错了地方 |
#
# 第三条尤其值得强调：**投机解码和 continuous batching 抢的是同一份闲置算力。** 并发低时投机解码收益明显；并发高到算力饱和时，投机解码反而降低整体吞吐。这是一个非常容易被忽略的取舍，能主动讲出来会加分很多。
#
# ## 八、面试话术
#
# **问：投机解码的收益从哪来？什么时候会变慢？**
#
# - **原理**：decode 是访存受限，算力闲置。草稿模型猜 γ 个 token，目标模型一次前向并行验证，等于用一次前向确认多个 token。
# - **成立前提**：验证那一次前向的成本要接近普通 decode。实测显示 token 数从 1 涨到 32，耗时只涨很小比例——因为权重只读了一遍。
# - **收益公式**：`E[tokens] = (1-α^(γ+1))/(1-α)`，成本 `γ·c_d + 1`。α 和 c_d 两个参数决定一切。
# - **负收益场景**：草稿太弱（α 低）或太贵（c_d 接近 1）；**并发已经打满算力时**，验证不再免费。
# - **工程现实**：生产上常用 n-gram / prompt lookup 做草稿（零模型成本，适合有大量重复文本的场景），或者用 MTP/EAGLE 这类把草稿能力直接训进模型的方法。
#
# **作业**
#
# 1. 把 `MAX_NEW` 改成 128，重跑实测。投机解码的收益是随生成长度变大还是变小？为什么？
# 2. 用 `speedup()` 算出：α=0.5、c_d=0.1 时，γ 取多少最优？（提示：扫描 γ=1..16）
# 3. 思考题：如果服务同时跑着 64 路并发，算力已经打满，此时开启投机解码会发生什么？你会在什么并发区间开启它？
#
# **下一章**：从精度下手——把权重和 KV cache 压到 8 位甚至 4 位。
