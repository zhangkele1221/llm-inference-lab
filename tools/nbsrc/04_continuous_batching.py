# %% [markdown]
# # 04 · vLLM 的调度循环：Continuous Batching
#
# 第 02 章证明了单个请求喂不饱 GPU。这一章解决下一个问题：**怎么把多个请求拼在一起跑。**
#
# 但这一章有个额外的目标：**让你学完之后打开 vLLM 源码能直接对上号。** 所以下面的类名、职责划分、主循环结构都刻意照搬了 vLLM V1。
#
# ## 三层架构（先记住这张图）
#
# ```
# EngineCore.step()                        ←→ vllm/v1/engine/core.py
#   ├─ Scheduler.schedule()                ←→ vllm/v1/core/sched/scheduler.py
#   │    决定这一轮跑哪些请求、各自几个 token      （不碰显存，不碰模型）
#   ├─ ModelRunner.execute_model()         ←→ vllm/v1/worker/gpu_model_runner.py
#   │    把排好的请求拼成 batch，跑模型，采样        （不碰调度策略）
#   └─ Scheduler.update_from_output()
#        把结果写回请求状态，处理完成与回收
# ```
#
# **这个职责划分是本章最值钱的东西。** 面试被问"vLLM 架构"，把这三层和各自的边界讲清楚，比背模块名有用得多。后面所有优化——chunked prefill、前缀缓存、抢占、投机解码——都是在这三步里插桩。
# %%
# @@SETUP@@
# %% [markdown]
# ## 一、先跑通一次，看清这个循环
#
# 引导单元里已经提供了 `Request`、`Scheduler`、`ModelRunner`、`EngineCore`。先做一次最小实验，把每一步的调度决策打出来。
# %%
model = build_model(block_size=4096)


def make_workload(n=32, prompt_len=64, lo=4, hi=65, vocab=50257, seed=0):
    """注意 prompt 是 Python list 不是张量——Request 持有的是 token id 序列，
    真正的张量由 ModelRunner 在跑模型时临时拼。这也是 vLLM 的做法。"""
    g = torch.Generator().manual_seed(seed)
    reqs = []
    for i in range(n):
        prompt = torch.randint(0, vocab, (prompt_len,), generator=g).tolist()
        max_tokens = int(torch.randint(lo, hi, (1,), generator=g).item())
        reqs.append(Request(f"req{i}", prompt, max_tokens))
    return reqs


# 6 条请求，batch 上限 4，看调度器怎么在 waiting 和 running 之间搬人
engine = EngineCore(model, Scheduler(max_num_seqs=4, max_num_batched_tokens=4096))
for r in make_workload(n=6, prompt_len=16, lo=2, hi=6, seed=1):
    engine.scheduler.add_request(r)

print(f"{'step':>5}{'running':>9}{'waiting':>9}   明细 (req: 已算token/prompt + 已生成)")
print("-" * 84)
while engine.scheduler.has_unfinished():
    engine.step()
    s = engine.scheduler
    detail = "  ".join(f"{r.request_id}:{r.num_computed_tokens}/{r.num_prompt_tokens}"
                       f"+{len(r.output_token_ids)}" for r in s.running)
    print(f"{s.step_id:>5}{len(s.running):>9}{len(s.waiting):>9}   {detail}")

print()
print("每个请求的最终状态：")
for r in engine.scheduler.finished:
    print(" ", r)

# %% [markdown]
# 观察这几点，它们就是 continuous batching 的全部内容：
#
# 1. **每轮结束就重新组批**。某个请求完成后立刻从 `running` 移除，`waiting` 里的新请求马上补进来——不需要等整批跑完。
# 2. **`num_computed_tokens` 一路往前推**。prefill 时它一次性涨到 prompt 长度；之后每轮 +1（decode）。这个字段是理解 vLLM 的钥匙。
# 3. **新请求在前几轮占用更多 token 预算**，因为要先把 prompt 算完。
# %% [markdown]
# ## 二、读懂 `EngineCore.step()`
#
# 主循环只有三步，但每一步的边界都很清晰：
#
# ```python
# def step(self):
#     sched_out = self.scheduler.schedule()                   # 1. 决定跑什么
#     sampled   = self.runner.execute_model(sched_out)        # 2. 跑模型
#     self.scheduler.update_from_output(sched_out, sampled)   # 3. 写回状态
# ```
#
# **为什么要把调度和执行分开？** 因为它们的变更频率完全不同：
#
# | 模块 | 多久改一次 | 改什么 |
# |---|---|---|
# | Scheduler | 很频繁 | 调度策略、优先级、抢占规则 |
# | ModelRunner | 很少改 | attention 后端、CUDA graph、量化 kernel |
#
# 揉在一起的话，调一个调度策略就要动模型执行代码，风险极高。分开之后，第 06 章要演示的 chunked prefill 只需要改 Scheduler 的一个预算参数。
#
# ### 一个刻意保留的差异
#
# `ModelRunner` 把请求分成了 `decode_reqs` 和 `prefill_reqs` 两组分别处理。真实 vLLM 是把它们**混在同一个 batch 里**跑的，这才是 chunked prefill 能成立的前提。
#
# 这里分开处理纯粹是为了让代码能读懂——本章的实验结论（调度层面的差异）不受影响。真实实现里混批需要 block table，那是第 05 章的内容。
# %% [markdown]
# ## 三、对照组：static batching
#
# 现在实现一个 static 版本的调度器。**注意：只改 `schedule()` 一个方法，执行和更新完全复用。** 这正是上面那个职责划分的价值——换调度策略不需要碰其他模块。
# %%
class TracedScheduler(Scheduler):
    """加一层记录，用来统计每步的批次占用情况。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trace = []

    def _record(self, out):
        self.trace.append({
            "step": self.step_id,
            "n_scheduled": len(out.scheduled_reqs),
            "n_running": len(self.running),
            "n_waiting": len(self.waiting),
            "tokens": sum(out.num_scheduled_tokens.values()),
        })
        return out

    def schedule(self):
        return self._record(super().schedule())


class StaticScheduler(TracedScheduler):
    """对照组：静态分组。一组全部跑完，才允许下一组进来。

    这就是传统 batching 的做法——像坐满一车人才发车的班车，
    而不是到站就上人下人的公交车。
    """

    def __init__(self, group_size=8, **kwargs):
        super().__init__(**kwargs)
        self.group_size = group_size

    def schedule(self):
        # 组内还有人就直接跑；整组清空了才开新的一组
        if not self.running and self.waiting:
            while self.waiting and len(self.running) < self.group_size:
                req = self.waiting.pop(0)
                req.status = "running"
                self.running.append(req)

        scheduled, num_tokens = [], {}
        budget = self.max_num_batched_tokens
        for req in list(self.running):
            if budget <= 0:
                break
            n = min(req.num_tokens_to_schedule(), budget)
            scheduled.append(req)
            num_tokens[req.request_id] = n
            budget -= n

        self.step_id += 1
        return self._record(SchedulerOutput(scheduled, num_tokens))


print("StaticScheduler 只重写了 schedule()，其余全部继承。")
print("这就是把调度和执行分开之后能拿到的东西。")
# %% [markdown]
# ## 四、对比实验
# %%
def run_engine(sched_cls, reqs, **kwargs):
    engine = EngineCore(model, sched_cls(**kwargs))
    for r in reqs:
        engine.scheduler.add_request(r)
    t0 = time.perf_counter()
    engine.run()
    dt = time.perf_counter() - t0
    return engine, dt


N, BATCH = 32, 8

static_engine, static_dt = run_engine(StaticScheduler, make_workload(N, seed=1),
                                      group_size=BATCH, max_num_batched_tokens=4096)
cont_engine, cont_dt = run_engine(TracedScheduler, make_workload(N, seed=1),
                                  max_num_seqs=BATCH, max_num_batched_tokens=4096)


def occupancy(engine, capacity):
    tr = engine.scheduler.trace
    return sum(t["n_running"] for t in tr) / len(tr) / capacity


total_tokens = sum(r.max_tokens for r in cont_engine.scheduler.finished)

print(f"{N} 条请求，batch 上限 {BATCH}，输出长度 4~64 随机\n")
print(f"{'调度方式':<22}{'总步数':>9}{'批次占用率':>12}{'耗时(s)':>10}{'吞吐(t/s)':>12}")
print("-" * 66)
print(f"{'static batching':<22}{static_engine.steps:>9}{occupancy(static_engine, BATCH):>11.1%}"
      f"{static_dt:>10.2f}{total_tokens / static_dt:>12,.0f}")
print(f"{'continuous batching':<22}{cont_engine.steps:>9}{occupancy(cont_engine, BATCH):>11.1%}"
      f"{cont_dt:>10.2f}{total_tokens / cont_dt:>12,.0f}")

# %% [markdown]
# ### 结果解读
#
# 两个指标的改善来自同一个机制：
#
# - **批次占用率**：static 会明显低于 continuous。因为一组里最短的输出 4 个 token、最长的 60 个，短的那条做完之后槽位就空着，直到整组跑完才换人。
# - **总步数**：每一步都要跑一次完整前向，步数少意味着端到端更快。continuous 的槽位几乎不空转，同样的工作量需要更少步数。
#
# **收益大小取决于输出长度的方差。** 如果所有请求输出长度都一样，static 的槽位永远不会空转，两者几乎没有差别。能说出这一点，说明你理解的是机制而不是结论。
#
# ### 代价
#
# 收益不是白来的，三个代价都要知道：
#
# 1. **延迟变得不可预测**：你不知道自己的请求会和谁拼在一起，p99 反而更难控制。线上必须用优先级或分池隔离。
# 2. **KV 管理复杂化**：batch 里每条序列进度不同、长度不同。本章的 `ModelRunner` 用"右填充 + 逐序列掩码"硬对齐，会浪费显存——这就是第 05 章 PagedAttention 要解决的问题。
# 3. **调度开销**：每轮都要重新组批，调度器本身也有成本。真实 vLLM 为此把调度逻辑写得很精细（优先级、抢占、token 预算分配）。
# %% [markdown]
# ## 五、对照表：这个 lab ↔ vLLM 源码
#
# 学到这里，这些名字你应该都能对应上了。路径以 vLLM V1 为准，版本间会移动，用 `rg` 定位最可靠。
#
# | 本章的东西 | vLLM 里的对应物 | 怎么找 | 差异 |
# |---|---|---|---|
# | `EngineCore.step()` | `EngineCore.step()` | `rg "def step" vllm/v1/engine/core.py` | 本 lab 同步执行，vLLM 异步 + 多进程 |
# | `Scheduler.schedule()` | `Scheduler.schedule()` | `rg "def schedule" vllm/v1/core/sched/` | vLLM 有优先级、抢占、`skipped_waiting` |
# | `SchedulerOutput` | `SchedulerOutput` | `vllm/v1/core/sched/output.py` | vLLM 还带 block 分配结果、preempted 列表 |
# | `Request.num_computed_tokens` | 同名字段 | `vllm/v1/request.py` | 语义完全一致 |
# | `Scheduler.running / waiting` | 同名字段 | `vllm/v1/core/sched/scheduler.py` | vLLM 还有 `skipped_waiting` |
# | `ModelRunner.execute_model()` | `GPUModelRunner.execute_model()` | `vllm/v1/worker/gpu_model_runner.py` | vLLM 要做 InputBatch、CUDA graph、metadata |
# | `max_num_batched_tokens` | **同名参数** | `vllm/config.py` → `SchedulerConfig` | 语义一模一样 |
# | `max_num_seqs` | **同名参数** | `vllm/config.py` → `SchedulerConfig` | 一模一样 |
#
# **两个参数的名字完全一致，这不是巧合**——它们就是你启动 vLLM 时能传的命令行参数。今天在 lab 里调它们观察到的现象，明天在真实服务上改它们会得到同样的结果。
# %% [markdown]
# ## 六、面试话术
#
# **问：continuous batching 相比 static batching 的收益来自哪？**
#
# 按这个顺序答：
#
# 1. **机制**：static 整批跑完才换人，短请求完成后槽位空转；continuous 每轮迭代重新组批，完成的立刻走、排队的立刻进。在 vLLM 里就是 `Scheduler` 的 `running` 和 `waiting` 两个队列在搬人。
# 2. **收益取决于什么**：输出长度分布的**方差**。方差越大，static 浪费越严重；长度整齐时两者几乎没差别。
# 3. **代价**：延迟不可预测、KV 管理复杂（催生了 PagedAttention）、调度本身有开销。
# 4. **架构视角（加分项）**：vLLM 把调度、显存、执行拆成 `Scheduler` / `KVCacheManager` / `ModelRunner` 三个模块，`EngineCore.step()` 是它们的胶水。这个划分让换调度策略不用碰模型执行代码——我在实验里只重写了一个 `schedule()` 方法就做出了 static batching 的对照组。
#
# 第 4 条是把"我懂概念"升级成"我读过源码"的关键。**能说出模块边界以及为什么这样划，比把机制复述一遍有说服力得多。**
#
# **作业**
#
# 1. 把 `make_workload` 的输出长度范围改成 `lo=60, hi=65`（差异很小），重跑实验。两种方式的差距消失了吗？
# 2. 把 `max_num_seqs` 从 8 改成 32（等于一次放进所有请求），continuous 会退化成什么？
# 3. 给 `Scheduler` 加一个优先级：让 `max_tokens` 小的请求优先调度。观察完成时间的分布变化。（提示：vLLM 里有 `priority` 参数和 `SchedulingPolicy`）
#
# **下一章**：解决本章留下的显存浪费——用分页的方式管理 KV cache。
