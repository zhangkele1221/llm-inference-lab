# %% [markdown]
# # 05 · Paged KV Cache 与前缀复用
#
# 上一章留了两个坑，这一章一起填掉：
#
# 1. **填充浪费**：一个 batch 里混进一条长序列，整个 batch 都要按最长的那条分配显存。
# 2. **重复计算**：大量请求共享同一段 prompt 前缀（系统提示、检索模板、few-shot 示例），却各自重算一遍。
#
# vLLM 的 PagedAttention 同时解决了这两个问题。这一章把它的核心机制亲手实现一遍。

# %%
# @@SETUP@@

# %% [markdown]
# ## 一、核心思路：把 KV cache 当虚拟内存管
#
# 操作系统怎么解决"进程需要连续内存但物理内存会碎片"的问题？**分页**——进程看到的是连续的虚拟地址，实际映射到任意物理页，靠页表翻译。
#
# PagedAttention 是同一个思路：
#
# | 操作系统 | PagedAttention |
# |---|---|
# | 物理页 | KV block（固定大小，比如 16 个 token） |
# | 页表 | block_table（记录这条序列用了哪些 block） |
# | 进程 | 一条请求序列 |
# | 共享库（多个进程共享代码页） | **前缀复用**（多条请求共享同一批 block） |
#
# 关键收益：序列的 KV 不再需要物理连续，也**不需要为了对齐而填充**。按需分配，用多少给多少。

# %%
class BlockManager:
    """KV cache 块分配器。这是 PagedAttention 里最关键的一个组件。"""

    def __init__(self, num_blocks, block_size):
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.free_blocks = list(range(num_blocks))
        self.tables = {}      # seq_id -> [block_id, ...]
        self.ref_count = {}   # block_id -> 被几条序列引用（前缀共享靠它）

    def blocks_needed(self, n_tokens):
        return math.ceil(n_tokens / self.block_size)

    def allocate(self, seq_id, n_tokens):
        """按需分配。已有够用就不动，不够才补。"""
        need = self.blocks_needed(n_tokens)
        have = len(self.tables.get(seq_id, []))
        if need > have:
            extra = need - have
            if extra > len(self.free_blocks):
                raise MemoryError(f"剩余 block 不足：需要 {extra}，只剩 {len(self.free_blocks)}")
            got = [self.free_blocks.pop() for _ in range(extra)]
            for b in got:
                self.ref_count[b] = 1
            self.tables.setdefault(seq_id, []).extend(got)
        return self.tables[seq_id]

    def free(self, seq_id):
        """释放 = 引用计数减一，减到 0 才真正归还。"""
        for b in self.tables.pop(seq_id, []):
            self.ref_count[b] -= 1
            if self.ref_count[b] == 0:
                self.free_blocks.append(b)

    def share_prefix(self, dst_id, src_id):
        """把 src 的 block_table 直接给 dst，引用计数加一——零拷贝的前缀复用。"""
        blocks = list(self.tables[src_id])
        self.tables[dst_id] = blocks
        for b in blocks:
            self.ref_count[b] += 1
        return blocks

    @property
    def used_blocks(self):
        return self.num_blocks - len(self.free_blocks)


bm = BlockManager(num_blocks=64, block_size=16)
bm.allocate("reqA", 100)
print(f"reqA 100 个 token → 占 {len(bm.tables['reqA'])} 个 block（112 token 的容量）")
bm.share_prefix("reqB", "reqA")
print(f"reqB 共享 reqA 前缀后，总占用仍是 {bm.used_blocks} 个 block（零拷贝）")
bm.free("reqA")
print(f"reqA 释放后仍占用 {bm.used_blocks} 个（因为 reqB 还引用着）")
bm.free("reqB")
print(f"reqB 也释放后占用 {bm.used_blocks} 个")

# %% [markdown]
# ## 二、分页省了多少：和"预留最大长度"对比
#
# 在没有分页的系统里，序列往往要**预留整个最大长度**的连续空间。分页则按需增长。差距有多大？

# %%
def utilization(strategy, lens, block_size=16, max_len=4096):
    ideal = sum(lens)
    if strategy == "reserve":      # 每条序列预留最大长度
        allocated = len(lens) * max_len
    elif strategy == "paged":      # 按需分配，向上取整到 block
        allocated = sum(math.ceil(L / block_size) * block_size for L in lens)
    else:
        allocated = ideal
    return ideal / allocated


workloads = {
    "8 条短请求 (100 token)": [100] * 8,
    "8 条长请求 (4K 上下文)": [4096] * 8,
    "长短混合 (100~4K)": [100, 200, 400, 800, 1600, 3200, 4096, 4096],
    "8 条刚起步 (10 token)": [10] * 8,
}

print(f"{'场景':<26}{'预留最大长度':>14}{'分页按需':>12}")
print("-" * 54)
for name, lens in workloads.items():
    print(f"{name:<26}{utilization('reserve', lens) * 100:>13.1f}%{utilization('paged', lens) * 100:>11.1f}%")

print()
print("注意最后一行：请求刚起步、只用了 10 个 token 时，预留式方案浪费了 99.8% 的空间。")
print("分页方案只浪费 block 内部的取整部分。线上大量请求同时处在不同阶段，")
print("这就是 PagedAttention 能把并发做上去的原因。")

# %% [markdown]
# ## 三、block size 怎么选
#
# 分页不是免费的：block 末尾用不满的部分就是**内部碎片**。block 越小碎片越少，但 block 表越长、寻址开销越大。

# %%
SEQ_LEN = 8192
print(f"以一条 {SEQ_LEN} token 的序列为例：\n")
print(f"{'block_size':>12}{'平均内部碎片':>18}{'block 表项数':>16}")
print("-" * 48)
for bs in [1, 4, 8, 16, 32, 64, 128, 256]:
    avg_frag = (bs - 1) / 2          # 长度均匀分布时的平均浪费
    print(f"{bs:>12}{avg_frag:>14.1f} token{SEQ_LEN // bs:>15}")

print()
print("怎么权衡：")
print("  block_size 太小 → block 表很长，每次 attention 要遍历更多块，索引开销上升")
print("  block_size 太大 → 内部碎片严重，短请求的显存被浪费")
print("  16 是主流默认值（vLLM 默认就是 16）：平均碎片 7.5 个 token，8K 序列的表 512 项，两边都还能接受")
print()
print("面试时能说出'这是碎片和寻址开销的折中，而且 kernel 实现对这个值有约束'，就高出一个层次了。")

# %% [markdown]
# ## 四、前缀复用：命中判定比你想象的严格
#
# 前缀缓存不是"内容相似就命中"，而是**按 block 做链式哈希，必须从第一个 block 起连续匹配**：
#
# ```
# hash(block 0) = H(-1,        tokens[0:B])
# hash(block 1) = H(hash(0),   tokens[B:2B])
# hash(block i) = H(hash(i-1), tokens[iB:(i+1)B])
# ```
#
# 为什么要链式？如果只哈希 block 内容，那么相同内容出现在**不同位置**时会被误判为可复用——但它前面的上下文不同，KV 完全不同，复用就会算错。

# %%
def block_hashes(token_ids, block_size):
    """链式哈希。末尾不足一个 block 的 token 不参与——因为它还会继续增长。"""
    tokens = list(token_ids)
    parent, out = -1, []
    for i in range(0, len(tokens) - block_size + 1, block_size):
        h = hash((parent, tuple(tokens[i:i + block_size])))
        out.append(h)
        parent = h
    return out


def matched_blocks(a, b, block_size):
    """返回两条序列从开头起连续匹配的 block 数。"""
    ha, hb = block_hashes(a, block_size), block_hashes(b, block_size)
    n = 0
    for x, y in zip(ha, hb):
        if x != y:
            break
        n += 1
    return n


BS = 16
BASE = list(range(1000, 1128))    # 128 token 的公共部分 = 8 个 block

base = BASE + [1, 2, 3, 4]
variants = {
    "同前缀，后缀不同": BASE + [9, 8, 7, 6],
    "前缀后追加变量": BASE + [555] + [1, 2, 3],
    "变量插在最开头": [777] + BASE,
    "变量插在第 64 token 后": BASE[:64] + [888] + BASE[64:],
}

print(f"block_size = {BS}，第一条序列共 {len(block_hashes(base, BS))} 个可缓存 block\n")
print(f"{'变体':<26}{'命中block':>10}{'判定':>16}")
print("-" * 52)
for name, seq in variants.items():
    m = matched_blocks(base, seq, BS)
    verdict = "高" if m >= 6 else ("低" if m > 0 else "完全失效")
    print(f"{name:<26}{m:>10}{verdict:>16}")

# %% [markdown]
# **这张表就是 prefix cache 的全部行为规律：**
#
# - 后缀不同不影响命中——这很好，检索到的文档本来就不一样。
# - 前缀后追加内容不影响命中——前面的 block 已经完整且固定了。
# - **变量插在最开头，命中率直接归零**。哪怕后面 100 个 token 完全相同，第一个 block 变了，链式哈希全断。
# - 变量插在中间，只有它之前的 block 能命中。
#
# 所以有一条工程铁律：**system prompt、指令模板、few-shot 放最前面，变量放最后面。**
#
# 上线前值得做一次审计：把线上 prompt 的各个字段按位置排一排，看哪个字段会让缓存整段失效。这通常是**改一行位置换来 30% 成本下降**的优化。

# %% [markdown]
# ## 五、验证前缀复用算得对
#
# 复用前缀的 KV 和整体重算，结果必须一致。这个验证不能省——复用错了不会报错，只会悄悄让输出变差。

# %%
model = build_model(block_size=4096)
P, S = 512, 128

prefix_ids = torch.randint(0, model.cfg.vocab_size, (1, P), device=DEVICE)
suffix_ids = torch.randint(0, model.cfg.vocab_size, (1, S), device=DEVICE)

logits_full, _ = model(torch.cat([prefix_ids, suffix_ids], dim=1))       # 整体重算
_, prefix_past = model(prefix_ids)
logits_reuse, _ = model(suffix_ids, past_kvs=prefix_past, pos_offset=P)  # 复用前缀 KV

diff = (logits_full[:, -1] - logits_reuse[:, -1]).abs().max().item()
same_token = torch.equal(logits_full[:, -1].argmax(-1), logits_reuse[:, -1].argmax(-1))

print(f"最后一位 logits 最大差异: {diff:.2e}")
print(f"argmax 结果一致        : {same_token}")
print()
print("差异应该在 1e-3 量级以下（浮点运算顺序不同导致的舍入），但 argmax 必须完全一致。")
print("如果 argmax 不一致，说明位置偏移或掩码算错了——这是复用前缀时最容易出的 bug。")

# %% [markdown]
# ## 六、复用能省多少计算

# %%
def prefill_cost(n_reqs, prefix_len, suffix_len, reuse=True):
    """需要计算的 token 数（prefill 计算量正比于 token 数）。"""
    if reuse:
        return prefix_len + n_reqs * suffix_len
    return n_reqs * (prefix_len + suffix_len)


N, P, Sfx = 8, 512, 64
no_reuse = prefill_cost(N, P, Sfx, reuse=False)
with_reuse = prefill_cost(N, P, Sfx, reuse=True)

print(f"{N} 条请求，每条 {P} token 共享前缀 + {Sfx} token 独立后缀\n")
print(f"不复用：{no_reuse:>6} token")
print(f"复用  ：{with_reuse:>6} token")
print(f"省下  ：{(1 - with_reuse / no_reuse) * 100:>5.1f}% 的 prefill 计算量")
print()
print("前缀越长、请求越密，收益越大。当共享前缀占到 prompt 的 90%（比如 RAG 里塞了整段检索模板），")
print("这就是数量级的差距。")

# %% [markdown]
# ## 七、面试话术
#
# **问：PagedAttention 解决什么问题？** 分三个层次答：
#
# 1. **消除外部碎片**：KV 不再要求物理连续，按 block 按需分配。
# 2. **消除对齐填充**：不同长度的序列能同 batch 跑，不需要补齐——上一章实测过，一条长序列混进来会让整个 batch 按最长分配。
# 3. **支持零拷贝前缀共享**：block 按引用计数共享，多条请求复用同一份前缀 KV。
#
# **问：block size 怎么选？** 内部碎片和寻址开销的折中。小了碎片少但表长、开销大；大了反过来。主流 16。
#
# **问：prefix cache 什么情况会完全失效？** **前缀第一个 block 就不同**。最常见原因是 prompt 把变量（用户 query、时间戳、请求 ID）放在了最前面。链式哈希必须从头连续匹配，第一个 block 断了后面全断。
#
# **作业**
#
# 1. 把 `BS` 改成 32 和 8，重跑哈希命中实验，观察"变量插在中间"那个用例的命中 block 数怎么变。
# 2. 回忆一个你线上真实遇到的场景：有没有哪段 prompt 因为字段顺序导致缓存失效？
# 3. 思考题：前缀共享用引用计数，那如果一条序列要**修改**共享 block 里的内容怎么办？（提示：写时复制 COW）
#
# **下一章**：长 prefill 会阻塞其他请求，怎么把它切碎混进 decode 里跑。

