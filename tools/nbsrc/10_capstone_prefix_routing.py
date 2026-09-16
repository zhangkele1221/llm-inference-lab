# %% [markdown]
# # 10 · 毕业项目：前缀感知路由
#
# 这是全课收尾，也是**可以直接写进简历的那个项目**。
#
# 问题：第 09 章证明了 prompt 结构影响单副本内的缓存命中。但线上一定是多副本的——请求被负载均衡器轮流传给各个副本，**同一个前缀在不同副本上各算一遍**，命中率被副本数稀释。
#
# 解法：在负载均衡层做**前缀感知路由**，把前缀相同的请求尽量送到同一个副本上。
#
# 这一章用真实计算搭出这个系统，并对比四种路由策略。
# %%
# @@SETUP@@
# %%
import time
from collections import Counter, OrderedDict
from statistics import mean, pstdev

model = build_model(block_size=4096)
PREFIX_LEN, SUFFIX_LEN = 256, 32
print(f"参数量 {model.n_params / 1e6:.1f}M，前缀 {PREFIX_LEN} token，后缀 {SUFFIX_LEN} token")
# %% [markdown]
# ## 一、副本：带前缀缓存的推理实例
#
# 每个副本有自己的显存预算（以能缓存的 token 数计）和一套 LRU 前缀缓存。
#
# > 这里的 `Replica` 相当于把第 04、05 章的 `EngineCore` + `KVCacheManager` 打包成一个可独立服务的实例。真实系统里每个副本就是一个独立的 vLLM 进程，本章用同一个模型对象模拟多个副本，是为了让单卡也能跑。
# %%
class Replica:
    def __init__(self, rid, model, capacity_tokens):
        self.rid = rid
        self.model = model
        self.capacity = capacity_tokens
        self.cache = OrderedDict()     # prefix_key -> (past, n_tokens)，按 LRU 淘汰
        self.used = 0
        self.hits = 0
        self.misses = 0
        self.computed_tokens = 0       # 实际算过的 token 数 —— 这就是 TTFT 的来源
        self.load = 0.0                # 衰减的负载计分

    def lookup(self, key):
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def insert(self, key, past, n_tokens):
        if key in self.cache:
            return
        self.cache[key] = (past, n_tokens)
        self.used += n_tokens
        while self.used > self.capacity and len(self.cache) > 1:
            _, (_, n) = self.cache.popitem(last=False)
            self.used -= n

    @torch.no_grad()
    def serve(self, prefix_ids, suffix_ids, key):
        """处理一条请求。命中缓存就只算后缀——这正是 TTFT 的差别所在。"""
        self.load = self.load * 0.9 + (PREFIX_LEN + SUFFIX_LEN)
        hit = self.lookup(key)

        if hit is not None:
            past, plen = hit
            self.model(suffix_ids, past_kvs=past, pos_offset=plen)
            self.computed_tokens += SUFFIX_LEN
            self.hits += 1
            return

        full = torch.cat([prefix_ids, suffix_ids], dim=1)
        _, past = self.model(full)
        self.computed_tokens += full.size(1)
        self.misses += 1
        # 因果注意力保证：前缀部分的 KV 不受后缀影响，可以直接切片出来缓存
        prefix_past = [(k[:, :, :PREFIX_LEN], v[:, :, :PREFIX_LEN]) for k, v in past]
        self.insert(key, prefix_past, PREFIX_LEN)


replicas = [Replica(i, model, capacity_tokens=2048) for i in range(3)]
print(f"启动 {len(replicas)} 个副本，每个前缀缓存容量 2048 token（约 8 条前缀）")
# %% [markdown]
# ## 二、构造工作负载：热点模板
#
# 真实线上流量从不均匀：少数几个 prompt 模板（系统提示、业务指令）占据绝大多数请求。这里用 Zipf 分布模拟。
# %%
def make_workload(n_requests=64, n_templates=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    templates = {
        t: torch.randint(0, model.cfg.vocab_size, (1, PREFIX_LEN), generator=g).to(DEVICE)
        for t in range(n_templates)
    }
    weights = torch.tensor([1.0 / (t + 1) for t in range(n_templates)])
    weights = weights / weights.sum()

    reqs = []
    for _ in range(n_requests):
        t = int(torch.multinomial(weights, 1, generator=g).item())
        suffix = torch.randint(0, model.cfg.vocab_size, (1, SUFFIX_LEN), generator=g).to(DEVICE)
        reqs.append({"key": t, "prefix": templates[t], "suffix": suffix})
    return reqs, templates


workload, templates = make_workload()
dist = Counter(r["key"] for r in workload)
print(f"共 {len(workload)} 条请求，模板分布：{dict(sorted(dist.items()))}")
print()
print("可以看到明显的长尾：模板 0 的请求数远多于模板 7。")
# %% [markdown]
# ## 三、四种路由策略
#
# | 策略 | 思路 | 预期 |
# |---|---|---|
# | 轮询 | 挨个分发 | 命中率最低，但绝对均衡 |
# | 最小负载 | 谁闲给谁 | 均衡好，但前缀被打散 |
# | 前缀亲和 | 同一前缀固定落点 | 命中率最高，但负载倾斜 |
# | 混合 | 亲和优先，过载则退避 | 折中 |
# %%
def route_round_robin(replicas, req, state):
    rep = replicas[state["i"] % len(replicas)]
    state["i"] += 1
    return rep


def route_least_load(replicas, req, state):
    return min(replicas, key=lambda r: r.load)


def route_prefix_affinity(replicas, req, state):
    """同一前缀永远落到同一个副本 → 命中率最大化，但可能倾斜。"""
    return replicas[req["key"] % len(replicas)]


def route_hybrid(replicas, req, state, threshold=1.6):
    """先看亲和副本；若它已过载（负载超过均值 threshold 倍），退而选最闲的。"""
    affine = replicas[req["key"] % len(replicas)]
    avg = mean(r.load for r in replicas) + 1e-6
    if affine.load <= avg * threshold:
        return affine
    return min(replicas, key=lambda r: r.load)


STRATEGIES = {
    "轮询": route_round_robin,
    "最小负载": route_least_load,
    "前缀亲和": route_prefix_affinity,
    "混合(亲和+负载)": route_hybrid,
}
# %% [markdown]
# ## 四、跑实验
# %%
def run(strategy_fn, workload):
    for r in replicas:
        r.cache.clear()
        r.used = r.hits = r.misses = r.computed_tokens = 0
        r.load = 0.0

    state = {"i": 0}
    t0 = time.perf_counter()
    for req in workload:
        rep = strategy_fn(replicas, req, state)
        rep.serve(req["prefix"], req["suffix"], req["key"])
    dt = time.perf_counter() - t0

    total = sum(r.hits + r.misses for r in replicas)
    hits = sum(r.hits for r in replicas)
    computed = sum(r.computed_tokens for r in replicas)
    loads = [r.computed_tokens for r in replicas]
    return {
        "耗时(s)": round(dt, 2),
        "命中率": f"{hits / total:.1%}",
        "总计算token": computed,
        "负载标准差": round(pstdev(loads)),
    }


ideal = len(workload) * SUFFIX_LEN + len(templates) * PREFIX_LEN
print(f"理论上限（每个模板只算一次前缀）: {ideal:,} token")

rows = {name: run(fn, workload) for name, fn in STRATEGIES.items()}

print()
print(f"{'策略':<18}{'命中率':>9}{'总计算token':>14}{'耗时(s)':>10}{'负载标准差':>12}")
print("-" * 64)
for name, r in rows.items():
    print(f"{name:<18}{r['命中率']:>9}{r['总计算token']:>14,}{r['耗时(s)']:>10.2f}{r['负载标准差']:>12,}")

print(f"\n理论上限是 {ideal:,} token，越接近说明缓存利用越充分。")
# %% [markdown]
# ## 五、结果解读
#
# 你应该会看到这样一组关系：
#
# | 策略 | 命中率 | 特点 |
# |---|---|---|
# | **轮询** | 最低 | 同一前缀被轮流送到所有副本，每个副本都要冷启动 |
# | **最小负载** | 也低 | 只看负载不看前缀，热点前缀依然被打散 |
# | **前缀亲和** | 最高 | 同前缀固定落点，命中率拉满，**但负载倾斜** |
# | **混合** | 接近亲和 | 在亲和与均衡之间取折中 |
#
# 这里藏着本项目的**核心权衡**，也是面试要讲的重点：
#
# > **缓存命中率要求"同前缀固定落点"，负载均衡要求"打散落点"，两者天然冲突。**
#
# 纯前缀亲和会把热点前缀的全部流量压到一个副本上，那个副本先过载；纯负载均衡则让缓存形同虚设。真实系统必须做折中。
#
# 你可以做一个更有说服力的分析：**扫描 `threshold` 参数**，画出"命中率"和"负载标准差"两条曲线，找出拐点。这条曲线就是你的项目结论。
# %% [markdown]
# ## 六、这个项目还能往哪走
#
# 想把它做成真正有分量的作品，下面每一条都可以继续做：
# %%
NEXT_STEPS = """
1. 副本故障与扩缩容
   副本挂了或扩容时，路由表怎么迁移？迁移导致缓存冷启动怎么预热？
   这是纯前缀亲和方案最脆弱的地方。

2. 热点前缀的复制策略
   当一个前缀热到单副本扛不住时，主动把它"灌"到多个副本上
   （每个副本都缓存一份），再对这组副本做负载均衡。
   这就把"固定落点"变成了"固定落点集合"。

3. 与前缀长度的联动
   前缀越长，缓存收益越大，越值得为它牺牲负载均衡；
   短前缀收益小，直接轮询即可。按前缀长度动态决定路由激进程度。

4. 真实框架对接
   这套逻辑在 K8s 生态里对应 Gateway API Inference Extension
   和 llm-d 的 prefix-aware routing；vLLM production stack 也有类似组件。
   把自研路由器换成它们的实现，对比效果和复杂度。

5. 观测指标
   路由层必须暴露：命中率、各副本负载分布、TTFT p50/p99、
   路由表大小、热点前缀 TOP-N。没有这些指标，线上出问题查不出来。
"""
print(NEXT_STEPS)
# %% [markdown]
# ## 七、写进简历的版本
#
# 这个项目的价值不只是"我做了个路由器"，而是它证明你能**从机制推导优化、再用数据验证**：
#
# > 针对多副本推理服务中前缀缓存命中率被负载均衡稀释的问题，设计并实现前缀感知路由层：通过 block 级前缀哈希做落点亲和，叠加负载感知退避避免热点倾斜，在前缀重复度 __% 的流量下将缓存命中率从 __% 提升至 __%，prefill 计算量下降 __%，等效 TTFT p99 下降 __%；并量化了"命中率与负载均衡的冲突边界"，给出按前缀长度动态调整路由激进程度的策略。
#
# 注意最后半句：**承认并量化权衡**，比只报一个漂亮百分比可信得多，也是资深工程师的表达方式。
#
# > 如果只在模拟环境验证，就老实写"基于模拟流量"。千万别写成线上数据——面试官一定会追问线上流量分布、灰度方案和回滚策略，答不上来就全盘可疑。
# %% [markdown]
# ## 八、课程结束，接下来做什么
#
# 十个 notebook 走完，你现在应该能够：
#
# - 白板画出一次推理迭代的完整数据流，并指出每一步的瓶颈在哪
# - 当场推算给定模型和卡型的最大并发
# - 解释 continuous batching / PagedAttention / prefix caching / chunked prefill / 投机解码各自的收益来源**和代价**
# - 拿到一个 vLLM 部署，按方法论定位吞吐问题
# - 识别不合理的 prompt 结构，并估算改动能省多少钱
#
# **下一步建议**：
#
# 1. 把线上真实的一段 prompt 和流量分布脱敏后，套用第 09 章的审计方法跑一遍。
# 2. 挑一个真正关心的优化点，把这里的模拟换成真实 vLLM 副本，产出带线上口径的数字。
# 3. 去给 vLLM 或 SGLang 提一个 PR。哪怕只是文档修正，走完一次完整的开源协作流程本身就有价值。
#
# 最后提醒一句：这个仓库里所有数字都是在你自己的机器上跑出来的。**面试时不要背这里的数字，要学会这里的推导方式**——推导方式才是别人拿不走的东西。
