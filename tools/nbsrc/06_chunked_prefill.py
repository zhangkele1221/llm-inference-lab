# %% [markdown]
# # 06 · Chunked Prefill 与尾延迟
#
# 前两章解决的是**吞吐**，这一章解决**延迟**——而且是延迟里最要命的那个指标：**尾延迟（p99）**。
#
# 场景很常见：服务正在稳定处理一批 decode 请求，突然来了一条 4K token 的长 prompt。
#
# ## 本章会推翻一个常见误解
#
# 很多人以为 chunked prefill 是 vLLM 的一个独立特性。不是。看完这章你会看到：
#
# > **它只是第 04 章那个 `Scheduler` 的 `max_num_batched_tokens` 参数调小了而已。**
#
# 不需要新代码路径，不需要新数据结构。长 prompt 之所以被切成多轮，是因为每轮的 token 预算装不下它——`num_computed_tokens` 这个字段会自动记录进度，下一轮接着算。
#
# 这就是第 04 章强调"读懂 `num_computed_tokens` 就抓住了主干"的原因。
# %%
# @@SETUP@@
# %%
model = build_model(block_size=4096)

# %% [markdown]
# ## 一、先看问题的严重程度
#
# 长 prefill 是一次无法中断的大矩阵乘法。在它执行期间 GPU 被它独占，其他请求只能等着。这就是**队头阻塞（head-of-line blocking）**。
# %%
dec_ids = torch.randint(0, model.cfg.vocab_size, (8, 128), device=DEVICE)
_, dec_past = model(dec_ids)
dec_next = torch.randint(0, model.cfg.vocab_size, (8, 1), device=DEVICE)
dec_ms = bench(lambda: model(dec_next, past_kvs=dec_past, pos_offset=128), warmup=3, iters=10)

print(f"单个 decode step（batch=8）: {dec_ms:.1f} ms\n")
for L in [256, 1024, 2048]:
    long_ids = torch.randint(0, model.cfg.vocab_size, (1, L), device=DEVICE)
    ms = bench(lambda: model(long_ids), warmup=2, iters=5)
    print(f"prefill {L:>5} token : {ms:>8.1f} ms   （是单个 decode step 的 {ms / dec_ms:>5.1f} 倍）")

print()
print("一条 2048 token 的 prefill，会让其他正在解码的请求多等几十到上百毫秒。")
print("用户视角就是'打字打到一半卡住了'。")
# %% [markdown]
# ## 二、问题出在 Scheduler 的哪一行
#
# 回顾第 04 章的 `Scheduler.schedule()`：
#
# ```python
# budget = self.max_num_batched_tokens
# ...
# n = min(req.num_tokens_to_schedule(), budget)   # ← 就是这一行
# ```
#
# 一个长 prompt 进来时，`num_tokens_to_schedule()` 返回 2048。如果预算是 4096，它一口气全拿走，这一轮就变成一次超长前向。
#
# **把预算调小到 256，同一行代码返回的就变成 256**——长 prompt 自动被切成多轮。没有第二个代码分支，没有开关。
#
# 下面用真实的 `EngineCore` 跑两遍，看同一行代码在不同预算下的行为。
# %%
def run_episode(budget, long_len=2048, n_decode=8, prompt_len=32, dec_steps=24):
    """跑一段真实调度：8 条请求已进入稳态 decode，中途插入一条长 prompt。

    返回 (每轮迭代耗时列表, 长请求在第几轮拿到第一个 token, 长请求对象)。
    """
    g = torch.Generator().manual_seed(7)
    sched = Scheduler(max_num_seqs=32, max_num_batched_tokens=budget)
    engine = EngineCore(model, sched)

    for i in range(n_decode):
        prompt = torch.randint(0, model.cfg.vocab_size, (prompt_len,), generator=g).tolist()
        sched.add_request(Request(f"dec{i}", prompt, dec_steps))

    # 先跑 3 轮，让这 8 条进入稳态 decode
    for _ in range(3):
        engine.step()

    # 现在插入长请求
    long_prompt = torch.randint(0, model.cfg.vocab_size, (long_len,), generator=g).tolist()
    long_req = Request("long", long_prompt, 1)
    sched.add_request(long_req)

    iters, long_ttft_step = [], None
    while sched.has_unfinished():
        t0 = time.perf_counter()
        engine.step()
        sync()
        iters.append((time.perf_counter() - t0) * 1000)
        if long_ttft_step is None and long_req.output_token_ids:
            long_ttft_step = len(iters)
    return iters, long_ttft_step, long_req


def pct(xs, q):
    s = sorted(xs)
    return s[min(len(s) - 1, int(q * len(s)))]


results = {}
for budget in [4096, 256]:
    results[budget] = run_episode(budget)

print("8 条稳态 decode 请求 + 1 条 2048 token 长 prompt 插入\n")
print(f"{'指标':<30}{'预算 4096':>14}{'预算 256':>14}")
print("-" * 60)
for label, fn in [
    ("每轮耗时 p50 (ms)", lambda it, tt: pct(it, 0.50)),
    ("每轮耗时 p95 (ms)", lambda it, tt: pct(it, 0.95)),
    ("每轮耗时 最大 (ms)", lambda it, tt: max(it)),
    ("总轮数", lambda it, tt: len(it)),
    ("长请求第几轮拿到 token", lambda it, tt: tt),
]:
    a = fn(results[4096][0], results[4096][1])
    b = fn(results[256][0], results[256][1])
    if isinstance(a, float):
        print(f"{label:<30}{a:>14.1f}{b:>14.1f}")
    else:
        print(f"{label:<30}{a:>14}{b:>14}")

# %% [markdown]
# ### 结果解读
#
# 你应该会看到：
#
# | 现象 | 说明 |
# |---|---|
# | **每轮耗时最大值大幅下降** | 预算 4096 时有一轮要吞下整个 2048 token 的 prefill；预算 256 时每轮都均匀 |
# | **长请求的 TTFT 变长** | 它要等好几轮才轮到算完，这是明确的代价 |
# | **总轮数增加** | 切得更碎，轮数更多，但每轮更快 |
#
# 这就是 chunked prefill 的本质：**它不是优化，是重新分配**。用长请求自己的 TTFT，换所有其他请求的尾延迟。
#
# 值不值得看业务：长请求占比低（比如 5%）而 decode 请求海量时，这笔交易非常划算——5% 的用户多等一点，95% 的用户不再卡顿。
# %% [markdown]
# ## 三、把调度轨迹打出来看
#
# 上面看的是结果，现在看**过程**——`Scheduler` 每轮到底排了多少 token。
# %%
def trace_budget(budget, long_len=1024, steps=14):
    g = torch.Generator().manual_seed(11)
    sched = Scheduler(max_num_seqs=32, max_num_batched_tokens=budget)
    engine = EngineCore(model, sched)
    for i in range(4):
        prompt = torch.randint(0, model.cfg.vocab_size, (32,), generator=g).tolist()
        sched.add_request(Request(f"d{i}", prompt, 40))
    sched.add_request(Request("LONG", torch.randint(0, model.cfg.vocab_size, (long_len,),
                                                  generator=g).tolist(), 1))

    print(f"  {'step':>4}{'本轮总token':>14}{'LONG 本轮':>11}{'LONG 进度':>22}")
    for _ in range(steps):
        if not sched.has_unfinished():
            break
        out = sched.schedule()
        n_long = out.num_scheduled_tokens.get("LONG", 0)
        lr = next(r for r in sched.running if r.request_id == "LONG")
        progress = f"{lr.num_computed_tokens}/{lr.num_prompt_tokens}"
        print(f"  {sched.step_id:>4}{sum(out.num_scheduled_tokens.values()):>14}"
              f"{n_long:>11}{progress:>22}")
        engine.runner.execute_model(out)
        sched.update_from_output(out, {})      # 只关心调度，不关心采样结果


print("预算 = 1024（长 prompt 一次装得下）")
trace_budget(1024)
print()
print("预算 = 128（同样的长 prompt 被自动切成多轮）")
trace_budget(128)

# %% [markdown]
# 两段轨迹唯一的区别就是预算数字，代码一行没改。
#
# 顺便注意 **`LONG 进度` 这一列**：它就是 `num_computed_tokens`，被切块之后它一格一格往前推，直到追上 prompt 长度才开始采样。这正是第 04 章说的——**读懂这个字段，chunked prefill 就不是一个独立机制了**。
# %% [markdown]
# ## 四、什么时候不该用它
#
# 面试官喜欢追问边界，这三个都是真实的：
#
# 1. **chunk 切得太小**：每个 chunk 的矩阵乘太小，GPU 算力利用率骤降，总吞吐反而变差。切分粒度要在"阻塞时间"和"算力效率"之间取平衡。
# 2. **显存压力大时**：分块让更多请求同时处于"进行中"，KV cache 峰值占用上升，可能触发抢占。用显存换延迟，账要算清楚。
# 3. **本来就延迟不敏感**：离线批量推理只关心吞吐，chunked prefill 只带来额外调度开销。
#
# 还有一个容易忽略的：**它和 prefix caching 有重叠**。如果 prompt 大部分能命中缓存，prefill 本来就短（第 05 章讲过 `num_computed_tokens` 直接被推上去），chunk 的意义就不大。两个优化不要重复投入。
#
# ### 一个真实的调参顺序
#
# 因为这两个优化有重叠，线上的调优顺序应该是：
#
# 1. **先修 prompt 结构**，把 prefix cache 命中率拉起来——这是免费的。
# 2. **再看尾延迟是否还需要改善**，需要才开 chunked prefill。
# 3. 调 `max_num_batched_tokens` 时，同时盯 `num_requests_waiting` 和 TTFT p99，别把吞吐调崩了。
# %% [markdown]
# ## 五、参数对照表
#
# | 本章的东西 | vLLM 里的对应物 | 说明 |
# |---|---|---|
# | `Scheduler.max_num_batched_tokens` | `SchedulerConfig.max_num_batched_tokens` | 名字和语义完全一致 |
# | `--enable-chunked-prefill` | 同名启动参数 | vLLM 里它是一个开关，开的本质是把预算调小 |
# | `Request.num_computed_tokens` | 同名字段 | 分块进度就记录在这里 |
# | `Scheduler.schedule()` 里的 `n = min(need, budget)` | `Scheduler` 里分配 token 预算的逻辑 | 就是这一行产生 chunked prefill |
#
# **关键认知**：vLLM 里 `enable_chunked_prefill` 这个开关之所以存在，是因为开启后调度策略会变化（比如不允许一个请求独占整个 batch），但底层的切分机制就是 token 预算，没有第二种实现。
# %% [markdown]
# ## 六、面试话术
#
# **问：chunked prefill 为什么能改善尾延迟？代价是什么？**
#
# - **问题**：prefill 是不可中断的一次大计算，长 prompt 独占 GPU，让同批正在解码的请求排队，表现为 TBT 尖刺。
# - **机制（这里要答准）**：它不需要新机制。`Scheduler` 每轮有一个 token 预算 `max_num_batched_tokens`，长 prompt 装不下就被自然切成多轮，`num_computed_tokens` 记录进度，下一轮接着算。
# - **收益**：把一次长阻塞摊成多次短阻塞，TBT p99 显著下降。
# - **代价**：长请求自身 TTFT 变长；chunk 过小降低算力效率；同时进行中的请求变多，KV 显存峰值上升。
# - **本质**：总计算量不变，是延迟在请求之间的**重新分配**。
#
# 最后那句"不是优化而是重新分配"，加上"它本质是 token 预算而不是新机制"，这两句加起来会让面试官确认你是真的读过源码，而不是看过几篇公众号。
#
# **作业**
#
# 1. 把预算从 256 改成 64 和 1024，重跑 episode，找出你机器上"尾延迟"和"总算力效率"的平衡点。
# 2. 把 `n_decode` 从 8 改成 32，长 prefill 的阻塞效应是变强还是变弱？为什么？
# 3. 思考题：如果长请求的用户体验很重要（比如付费用户），设计什么机制既保住他的 TTFT 又保住其他人的尾延迟？（提示：vLLM 的 `priority` 参数 + 调度优先级 + 抢占）
#
# **下一章**：换个方向提速——用一个小模型给大模型"打草稿"，也就是投机解码。
