# %% [markdown]
# # 05 · PagedAttention 与前缀缓存
#
# 第 04 章留了一个坑：`ModelRunner` 为了把不同进度的序列拼进一个 batch，必须把 KV **右填充到同一长度**。一个 batch 里混进一条长序列，所有序列都要按最长的那条分配显存。
#
# 这一章补上 vLLM 的 KV 管理层，同时解决另一个问题：**大量请求共享同一段 prompt 前缀，却各自重算一遍。**
#
# ## 架构位置
#
# 第 04 章说过调度和执行是分开的，现在补上第三块：
#
# ```
# EngineCore.step()
#   ├─ Scheduler.schedule()                   决定跑什么        ← 第 04 章
#   │    └─ KVCacheManager.allocate_slots()   决定 KV 放哪       ← 本章
#   ├─ ModelRunner.execute_model()            跑模型
#   └─ Scheduler.update_from_output()
# ```
#
# 本章的类对应关系：
#
# | 本章 | vLLM 源码 |
# |---|---|
# | `KVCacheBlock` | `vllm/v1/core/kv_cache_utils.py` |
# | `BlockPool` | `vllm/v1/core/block_pool.py` |
# | `KVCacheManager` | `vllm/v1/core/kv_cache_manager.py` |
# | `hash_block_tokens()` | `vllm/v1/core/kv_cache_utils.py` |
# %%
# @@SETUP@@
# %%
from collections import deque

print("本章只用引导单元里的 MiniGPT，调度部分用不到——因为 KV 管理层和调度层是解耦的。")
# %% [markdown]
# ## 一、核心思路：把 KV cache 当虚拟内存管
#
# 操作系统怎么解决"进程需要连续内存，但物理内存会碎片"？**分页**——进程看到的是连续虚拟地址，实际映射到任意物理页，靠页表翻译。
#
# PagedAttention 是同一个思路：
#
# | 操作系统 | PagedAttention |
# |---|---|
# | 物理页 | KV block（固定大小，vLLM 默认 16 个 token） |
# | 页表 | `block_table`（这条序列用了哪些 block） |
# | 进程 | 一条请求序列 |
# | 共享库（多进程共享代码页） | **前缀缓存**（多条请求共享同一批 block） |
#
# 关键收益：序列的 KV 不再需要物理连续，也**不需要为了对齐而填充**。按需分配，用多少给多少。
# %% [markdown]
# ## 二、KVCacheBlock 与链式哈希
# %%
class KVCacheBlock:
    """对应 vllm/v1/core/kv_cache_utils.py 的 KVCacheBlock。

    一个物理 block 只需要这几个字段：
      · ref_cnt    —— 被几条序列引用（前缀共享靠它）
      · block_hash —— 链式哈希值，None 表示还没填满、不可复用
    """

    __slots__ = ("block_id", "ref_cnt", "block_hash")

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_cnt = 0
        self.block_hash = None

    def __repr__(self):
        h = "----" if self.block_hash is None else f"{self.block_hash % 10000:04d}"
        return f"Block#{self.block_id}(ref={self.ref_cnt},hash={h})"


def hash_block_tokens(parent_block_hash, curr_block_token_ids):
    """对应 vllm/v1/core/kv_cache_utils.py 的 hash_block_tokens。

    链式哈希：哈希值 = H(前一个 block 的哈希, 当前 block 的 token)。

    为什么必须把 parent 算进去？如果只哈希当前 block 的内容，那么**相同内容出现在
    不同位置**时会被误判成可复用——但它们前面的上下文不同，KV 完全不同，
    一旦复用就会算错。这是链式哈希存在的唯一理由。
    """
    return hash((parent_block_hash, tuple(curr_block_token_ids)))


same = [1, 2, 3, 4]
h0 = hash_block_tokens(-1, same)
h1 = hash_block_tokens(h0, same)
print("链式哈希演示：")
print(f"  第一次出现的 hash : {h0 % 10000:04d}")
print(f"  换个位置再出现    : {h1 % 10000:04d}   ← 不同，所以不会被误复用")
# %% [markdown]
# ## 三、BlockPool：物理 block 的分配与回收
# %%
class BlockPool:
    """对应 vllm/v1/core/block_pool.py 的 BlockPool。

    管理全部物理 block：分配、回收、以及前缀哈希表。

    与真实 vLLM 的差异：vLLM 的 free blocks 用 FreeKVCacheBlockQueue（双向链表，
    O(1) 增删），并把"有哈希但没人引用"的 block 单独放进淘汰队列，
    让前缀缓存和回收可以同时成立。这里用 deque + 覆写时摘除哈希简化实现，
    语义等价，在讲解场景下行为一致。
    """

    def __init__(self, num_blocks, block_size):
        self.block_size = block_size
        self.blocks = [KVCacheBlock(i) for i in range(num_blocks)]
        self.free_blocks = deque(b.block_id for b in self.blocks)
        # 前缀哈希 → block，这就是 prefix cache 的本体
        self.cached_block_hash_to_block = {}

    def get_new_blocks(self, n):
        """对应 vLLM 的 get_new_blocks。"""
        if n > len(self.free_blocks):
            raise MemoryError(f"block 不足：需要 {n}，剩余 {len(self.free_blocks)}")
        ids = [self.free_blocks.popleft() for _ in range(n)]
        for i in ids:
            b = self.blocks[i]
            if b.block_hash is not None:
                # 这个 block 原来缓存着别的前缀，现在要被覆写，从缓存表里摘掉
                self.cached_block_hash_to_block.pop(b.block_hash, None)
                b.block_hash = None
            b.ref_cnt += 1
        return ids

    def free_blocks_of(self, block_ids):
        """对应 vLLM 的 free_blocks：引用计数减到 0 才真正回到空闲队列。"""
        for i in block_ids:
            b = self.blocks[i]
            b.ref_cnt -= 1
            if b.ref_cnt == 0:
                self.free_blocks.append(i)

    def cache_full_block(self, block_id, block_hash):
        """一个 block 被填满后注册进前缀哈希表，后续请求才可能命中。"""
        b = self.blocks[block_id]
        b.block_hash = block_hash
        self.cached_block_hash_to_block[block_hash] = b

    @property
    def num_free(self):
        return len(self.free_blocks)


pool = BlockPool(num_blocks=32, block_size=16)
print(f"启动一个 {len(pool.blocks)} 个 block、每块 16 token 的 KV 池"
      f"（总容量 {len(pool.blocks) * 16} token）")
# %% [markdown]
# ## 四、KVCacheManager：前缀命中就是把 `num_computed_tokens` 推上去
#
# **这是本章最重要的一段。** 第 04 章反复强调 `num_computed_tokens` 是理解 vLLM 的钥匙，现在你会看到它怎么和前缀缓存配合：
#
# 请求进来时先查前缀缓存。命中 N 个 block，就把 `num_computed_tokens` 直接设成 `N × block_size`——**这些 token 一个都不用算**，剩下的部分才进入正常的 prefill。
# %%
class KVCacheManager:
    """对应 vllm/v1/core/kv_cache_manager.py 的 KVCacheManager。

    注意职责边界：它不决定"跑什么"（那是 Scheduler 的事），
    只回答两个问题——KV 放哪里、能复用多少。
    """

    def __init__(self, block_pool):
        self.block_pool = block_pool
        self.block_tables = {}          # request_id -> [block_id, ...]

    def _match_prefix_blocks(self, req):
        """逐块做链式哈希比对，返回从开头起能连续复用的 block 对象列表。

        注意末尾不满一个 block 的 token 永远匹配不上——因为它还会继续增长，
        这一块的内容还会变。这就是"最后一块不可复用"的根本原因。
        """
        bs = self.block_pool.block_size
        tokens = req.all_token_ids()
        parent_hash, matched = -1, []
        for start in range(0, len(tokens) // bs * bs, bs):
            h = hash_block_tokens(parent_hash, tokens[start:start + bs])
            blk = self.block_pool.cached_block_hash_to_block.get(h)
            if blk is None:
                break
            matched.append(blk)
            parent_hash = h
        return matched

    def attach_prefix_cache(self, req):
        """前缀缓存命中时的动作：零拷贝共享 block + 推进 num_computed_tokens。"""
        matched = self._match_prefix_blocks(req)
        if not matched:
            return 0
        for blk in matched:
            blk.ref_cnt += 1                       # 共享，不是复制
        self.block_tables[req.request_id] = [b.block_id for b in matched]
        req.num_computed_tokens = len(matched) * self.block_pool.block_size
        return len(matched)

    def allocate_slots(self, req, num_tokens):
        """为本轮要计算的 num_tokens 个 token 补齐所需 block。"""
        bs = self.block_pool.block_size
        need = math.ceil((req.num_computed_tokens + num_tokens) / bs)
        table = self.block_tables.setdefault(req.request_id, [])
        if need > len(table):
            table.extend(self.block_pool.get_new_blocks(need - len(table)))
        return table

    def cache_blocks(self, req):
        """把已经填满的 block 注册进前缀缓存，供后续请求复用（对应 cache_full_blocks）。"""
        bs = self.block_pool.block_size
        tokens = req.all_token_ids()
        table = self.block_tables.get(req.request_id, [])
        parent_hash, n_cached = -1, 0
        for idx, block_id in enumerate(table):
            start = idx * bs
            if start + bs > req.num_computed_tokens:
                break                      # 还没填满，不能缓存
            h = hash_block_tokens(parent_hash, tokens[start:start + bs])
            self.block_pool.cache_full_block(block_id, h)
            parent_hash = h
            n_cached += 1
        return n_cached

    def free_request(self, req):
        """请求结束后释放引用，计数归零的 block 回到空闲队列。"""
        self.block_pool.free_blocks_of(self.block_tables.pop(req.request_id, []))


kv_manager = KVCacheManager(pool)
# %% [markdown]
# ## 五、亲眼看到前缀命中跳过了多少计算
# %%
PREFIX_LEN, SUFFIX_LEN = 256, 32
prefix_tokens = list(range(10000, 10000 + PREFIX_LEN))


def make_request(rid, suffix_start):
    prompt = prefix_tokens + list(range(suffix_start, suffix_start + SUFFIX_LEN))
    return Request(rid, prompt, 4)


# 请求 A：冷启动，全量 prefill，然后把自己的 block 缓存起来
req_a = make_request("A", 20000)
print(f"请求 A 入场: 命中 0 个 block，需要计算 {req_a.num_tokens_to_schedule()} token")
kv_manager.allocate_slots(req_a, req_a.num_tokens_to_schedule())
req_a.num_computed_tokens = req_a.num_prompt_tokens       # 模拟"算完了"
n_cached = kv_manager.cache_blocks(req_a)
print(f"  A 算完后缓存了 {n_cached} 个 block，缓存表大小 {len(pool.cached_block_hash_to_block)}")
print(f"  A 的 block_table: {kv_manager.block_tables['A']}")

# 请求 B：共享同一段前缀，应该直接命中
req_b = make_request("B", 30000)
hit = kv_manager.attach_prefix_cache(req_b)
print()
print(f"请求 B 入场: 命中 {hit} 个 block")
print(f"  num_computed_tokens 被直接推进到 {req_b.num_computed_tokens}/{req_b.num_prompt_tokens}")
print(f"  还需要计算的 token 数: {req_b.num_tokens_to_schedule()}   ← 只剩后缀")

# 请求 C：前缀完全不同，命中 0
req_c = Request("C", list(range(70000, 70000 + PREFIX_LEN)), 4)
hit_c = kv_manager.attach_prefix_cache(req_c)
print()
print(f"请求 C（前缀完全不同）: 命中 {hit_c} 个 block，仍需计算 {req_c.num_tokens_to_schedule()} token")

print()
print("看 B 和 C 的对比——这就是前缀缓存的全部收益：")
print("  命中的 token 一个都不用算，num_computed_tokens 被直接推上去。")
print("  在 vLLM 里这个字段会被 Scheduler 读走，用来算本轮该给这个请求排多少 token。")
# %% [markdown]
# ### 引用计数：共享但不复制
# %%
shared = kv_manager.block_tables["B"]
print("请求 A 和 B 共享的 block：")
for bid in shared[:4]:
    print(f"  {pool.blocks[bid]}")
print("  ...")

kv_manager.free_request(req_a)
print()
print("释放请求 A 之后：")
for bid in shared[:4]:
    print(f"  {pool.blocks[bid]}   ← B 还引用着，没有回到空闲队列")

kv_manager.free_request(req_b)
print()
print("释放请求 B 之后：")
for bid in shared[:4]:
    print(f"  {pool.blocks[bid]}   ← 引用计数归零，块回到空闲队列")

print(f"\n空闲 block 数: {pool.num_free}")
print()
print("引用计数是前缀共享能成立的关键：多个请求读同一份物理块，谁都不复制。")
print("一旦某条序列要往里写（分叉后产生新 token），就需要写时复制（COW）——")
print("vLLM 里的做法是 block 被覆写前先从缓存哈希表里摘除。")
# %% [markdown]
# ## 六、验证：复用前缀算出来的结果必须一致
#
# 上面演示的是**账本**。账本对了不代表算得对——复用前缀的 KV 和整体重算，输出必须一致。这个验证不能省，因为复用错了不会报错，只会悄悄让输出变差。
# %%
model = build_model(block_size=4096)
P, S = 512, 128

prefix_ids = torch.randint(0, model.cfg.vocab_size, (1, P), device=DEVICE)
suffix_ids = torch.randint(0, model.cfg.vocab_size, (1, S), device=DEVICE)

logits_full, _ = model(torch.cat([prefix_ids, suffix_ids], dim=1))       # 整体重算
_, prefix_past = model(prefix_ids)
logits_reuse, _ = model(suffix_ids, past_kvs=prefix_past, pos_offset=P)  # 复用前缀 KV

diff = (logits_full[:, -1] - logits_reuse[:, -1]).abs().max().item()
same = torch.equal(logits_full[:, -1].argmax(-1), logits_reuse[:, -1].argmax(-1))

print(f"最后一位 logits 最大差异: {diff:.2e}")
print(f"argmax 结果一致        : {same}")
print()
print("差异应该在 1e-3 量级以下（浮点运算顺序导致的舍入），但 argmax 必须完全一致。")
print("如果不一致，说明位置偏移或掩码算错了——这是复用前缀时最容易出的 bug。")
# %% [markdown]
# ## 七、block size 怎么选
#
# 分页不是免费的：block 末尾用不满的部分就是**内部碎片**。block 越小碎片越少，但 block 表越长、寻址开销越大。
# %%
SEQ_LEN = 8192
print(f"以一条 {SEQ_LEN} token 的序列为例：\n")
print(f"{'block_size':>12}{'平均内部碎片':>18}{'block 表项数':>16}")
print("-" * 48)
for bs in [1, 4, 8, 16, 32, 64, 128, 256]:
    print(f"{bs:>12}{(bs - 1) / 2:>14.1f} token{SEQ_LEN // bs:>15}")

print()
print("怎么权衡：")
print("  block_size 太小 → block 表很长，attention 要遍历更多块，索引开销上升")
print("  block_size 太大 → 内部碎片严重，短请求的显存被浪费")
print("  16 是 vLLM 的默认值：平均碎片 7.5 个 token，8K 序列表长 512，两边都还能接受")
print()
print("面试时能说出'这是碎片和寻址开销的折中，而且 attention kernel 对这个值有约束'，")
print("就明显高出一个层次。")
# %% [markdown]
# ## 八、分页到底省了多少
# %%
def utilization(strategy, lens, block_size=16, max_len=4096):
    ideal = sum(lens)
    if strategy == "reserve":      # 每条序列预留最大长度
        allocated = len(lens) * max_len
    elif strategy == "padded":     # 第 04 章的填充方案：按 batch 内最长对齐
        allocated = len(lens) * max(lens)
    else:                          # 分页按需
        allocated = sum(math.ceil(L / block_size) * block_size for L in lens)
    return ideal / allocated


workloads = {
    "8 条短请求 (100)": [100] * 8,
    "8 条长请求 (4096)": [4096] * 8,
    "长短混合 (100~4K)": [100, 200, 400, 800, 1600, 3200, 4096, 4096],
    "8 条刚起步 (10)": [10] * 8,
}

print(f"{'场景':<24}{'预留最大长度':>14}{'对填充(第04章)':>16}{'分页按需':>12}")
print("-" * 68)
for name, lens in workloads.items():
    print(f"{name:<24}{utilization('reserve', lens) * 100:>13.1f}%"
          f"{utilization('padded', lens) * 100:>15.1f}%{utilization('paged', lens) * 100:>11.1f}%")

print()
print("中间那列就是第 04 章 ModelRunner 实际在做的事——为了对齐而填充。")
print("请求刚起步、只用了 10 个 token 时，填充方案浪费 99.8%，分页只损失 block 内取整。")
# %% [markdown]
# ## 九、前缀命中的规律（必背）
# %%
def block_hashes(tokens, bs):
    """整条序列的链式哈希列表。末尾不足一个 block 的 token 不参与。"""
    parent, out = -1, []
    for i in range(0, len(tokens) - bs + 1, bs):
        h = hash_block_tokens(parent, tokens[i:i + bs])
        out.append(h)
        parent = h
    return out


def matched_blocks(a, b, bs):
    """两条序列从开头起连续匹配的 block 数。"""
    n = 0
    for x, y in zip(block_hashes(a, bs), block_hashes(b, bs)):
        if x != y:
            break
        n += 1
    return n


BS = 16
BASE = list(range(1000, 1128))    # 128 token 公共部分 = 8 个 block
base = BASE + [1, 2, 3, 4]

variants = {
    "同前缀，后缀不同": BASE + [9, 8, 7, 6],
    "前缀后追加变量": BASE + [555] + [1, 2, 3],
    "变量插在最开头": [777] + BASE,
    "变量插在第 64 token 后": BASE[:64] + [888] + BASE[64:],
}

print(f"对照组有 {len(block_hashes(base, BS))} 个可缓存 block"
      f"（末尾不足一块的 4 个 token 不算）\n")
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
# 上线前值得做一次审计：把线上 prompt 的每个字段按位置排一排，看哪个字段会让缓存整段失效。这通常是**改一行位置换来 30% 成本下降**的优化。
# %% [markdown]
# ## 十、API 对照表
#
# | 本章的方法 | vLLM 里的对应物 | 怎么找 |
# |---|---|---|
# | `KVCacheManager.attach_prefix_cache()` | `get_computed_blocks()` + `allocate_slots()` | `rg "def get_computed_blocks" vllm/v1/core/kv_cache_manager.py` |
# | `KVCacheManager.allocate_slots()` | `allocate_slots()` | 同上文件 |
# | `KVCacheManager.cache_blocks()` | `cache_full_blocks()` | `rg "def cache_full_blocks" vllm/v1/core/block_pool.py` |
# | `KVCacheManager.free_request()` | `free()` | `vllm/v1/core/kv_cache_manager.py` |
# | `BlockPool.get_new_blocks()` | `get_new_blocks()` | `vllm/v1/core/block_pool.py` |
# | `BlockPool.free_blocks_of()` | `free_blocks()` | 同上 |
# | `BlockPool.cached_block_hash_to_block` | 同名字段 | 同上 |
# | `KVCacheBlock.ref_cnt / block_hash` | 同名字段 | `vllm/v1/core/kv_cache_utils.py` |
# | `hash_block_tokens()` | `hash_block_tokens()` | `vllm/v1/core/kv_cache_utils.py` |
# | `BlockPool.free_blocks`（deque） | `FreeKVCacheBlockQueue` | `vllm/v1/core/kv_cache_utils.py` |
#
# **字段名和方法名几乎完全一致，这是故意的。** 你现在打开 `block_pool.py`，看到的应该是一堆熟悉的东西。
# %% [markdown]
# ## 十一、面试话术
#
# **问：PagedAttention 解决什么问题？** 分三个层次答：
#
# 1. **消除外部碎片**：KV 不再要求物理连续，按 block 按需分配。
# 2. **消除对齐填充**：不同长度的序列能进同一个 batch，不需要补齐。第 04 章的填充方案在请求刚起步时浪费 99.8%，分页只损失 block 内取整。
# 3. **支持零拷贝前缀共享**：block 按引用计数共享，多条请求读同一份物理块。
#
# **问：前缀缓存命中之后发生了什么？**
#
# 答：请求的 `num_computed_tokens` 被直接推进到命中长度，这些 token 一个都不用算。Scheduler 下一轮读这个字段，就知道只该给剩下那点后缀排 token。**命中和不命中，在 vLLM 里体现为同一个字段的不同取值，而不是两套代码路径。**
#
# **问：block size 怎么选？** 内部碎片和寻址开销的折中。小了碎片少但表长、开销大；大了反过来。vLLM 默认 16。
#
# **问：prefix cache 什么情况会完全失效？** **前缀第一个 block 就不同**。最常见原因是 prompt 把变量（用户 query、时间戳、请求 ID）放在最前面。链式哈希必须从头连续匹配，第一个 block 断了后面全断。
#
# **作业**
#
# 1. 把 `BS` 改成 32 和 8，重跑命中实验，观察"变量插在中间"那个用例的命中 block 数怎么变。
# 2. 给 `BlockPool` 加一个 LRU 淘汰：`get_new_blocks` 找不到空闲块时，优先淘汰哈希表中引用计数为 0 的 block。想想这和 vLLM 的 eviction queue 是不是同一个东西。
# 3. 思考题：两条序列共享了一个 block，其中一条分叉产生了新 token，怎么处理？（提示：写时复制 COW，vLLM 里是覆写前先摘除缓存哈希）
#
# **下一章**：把 `Scheduler` 的 token 预算调小，看 chunked prefill 怎么自然浮现出来。
