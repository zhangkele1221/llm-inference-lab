# %% [markdown]
# # 06 · Chunked Prefill 与尾延迟
#
# 前两章解决**吞吐**问题。这一章解决**延迟**，而且是延迟里最要命的那个指标：**尾延迟（p99）**。
#
# 场景很常见：服务正在稳定处理一批 decode 请求，突然来了一条 4K token 的长 prompt。会发生什么？

# %%
# @@SETUP@@

# %% [markdown]
# ## 一、先看问题的严重程度
#
# 长 prefill 是一次无法中断的大矩阵乘法。在它执行期间 GPU 被它独占，其他请求只能等着。这就是**队头阻塞（head-of-line blocking）**。

# %%
model = build_model(block_size=4096)

# 一个 decode step（batch=8，上下文 128）
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
# ## 二、Chunked Prefill 的做法
#
# 思路：**把长 prefill 切成若干小块，每轮迭代只算一块，和该轮的 decode 请求拼在一起执行。**
#
# 关键认知：**这不会让总计算量减少。** 它只是把一次很长的阻塞，摊成多次很短的阻塞。
#
# 实现上，借助 `past_kvs` 和 `pos_offset` 就能连续分块：

# %%
@torch.no_grad()
def chunked_prefill(model, prompt, chunk_size):
    """把长 prompt 分块喂进去，每块复用前面累积的 KV。"""
    past, logits = None, None
    for start in range(0, prompt.size(1), chunk_size):
        piece = prompt[:, start:start + chunk_size]
        logits, past = model(piece, past_kvs=past, pos_offset=start)
    return logits, past


# 先验证分块和整段跑结果一致
prompt = torch.randint(0, model.cfg.vocab_size, (1, 1024), device=DEVICE)
whole_logits, _ = model(prompt)
chunk_logits, _ = chunked_prefill(model, prompt, chunk_size=256)

print(f"整段与分块的最后一位 logits 最大差异: "
      f"{(whole_logits[:, -1] - chunk_logits[:, -1]).abs().max().item():.2e}")
print(f"argmax 一致: {torch.equal(whole_logits[:, -1].argmax(-1), chunk_logits[:, -1].argmax(-1))}")

# %% [markdown]
# ## 三、真实调度对比
#
# 跑一个真实的调度过程：8 条请求正在稳定解码，中途一条 2048 token 的长 prompt 插入。

# %%
@torch.no_grad()
def episode(model, chunked, n_decode=8, dec_ctx=128, steps=24, long_len=2048, chunk=256):
    """返回 (每轮迭代耗时列表, 长请求 TTFT, 总耗时)。"""
    dec_ids = torch.randint(0, model.cfg.vocab_size, (n_decode, dec_ctx), device=DEVICE)
    logits, dec_past = model(dec_ids)
    dec_pos = dec_ctx
    dec_next = logits[:, -1].argmax(-1, keepdim=True)

    long_prompt = torch.randint(0, model.cfg.vocab_size, (1, long_len), device=DEVICE)
    n_chunks = math.ceil(long_len / chunk)
    chunk_idx = 0
    long_past = None
    long_ttft = None

    t_start = time.perf_counter()
    t_prev = t_start
    iters = []

    for step in range(steps):
        if chunked:
            # 每轮：一个 prefill 小块 + 一次 decode，混在一起跑
            if chunk_idx < n_chunks:
                start = chunk_idx * chunk
                piece = long_prompt[:, start:start + chunk]
                _, long_past = model(piece, past_kvs=long_past, pos_offset=start)
                chunk_idx += 1
                if chunk_idx == n_chunks:
                    long_ttft = time.perf_counter() - t_start
            logits, dec_past = model(dec_next, past_kvs=dec_past, pos_offset=dec_pos)
            dec_pos += 1
            dec_next = logits[:, -1].argmax(-1, keepdim=True)
        else:
            # 第一轮就把整段长 prefill 跑完，其他请求全部排队等它
            if step == 0:
                _, long_past = model(long_prompt)
                long_ttft = time.perf_counter() - t_start
            logits, dec_past = model(dec_next, past_kvs=dec_past, pos_offset=dec_pos)
            dec_pos += 1
            dec_next = logits[:, -1].argmax(-1, keepdim=True)

        sync()
        now = time.perf_counter()
        iters.append((now - t_prev) * 1000)
        t_prev = now

    return iters, long_ttft, time.perf_counter() - t_start


def pct(xs, q):
    s = sorted(xs)
    return s[min(len(s) - 1, int(q * len(s)))]


base_iters, base_ttft, base_total = episode(model, chunked=False)
chunk_iters, chunk_ttft, chunk_total = episode(model, chunked=True)

print("8 条稳定解码请求 + 1 条 2048 token 长 prompt 插入，共 24 轮迭代\n")
print(f"{'指标':<26}{'整段 prefill':>16}{'chunked prefill':>18}")
print("-" * 60)
print(f"{'解码请求 TBT p50 (ms)':<26}{pct(base_iters, 0.50):>16.1f}{pct(chunk_iters, 0.50):>18.1f}")
print(f"{'解码请求 TBT p95 (ms)':<26}{pct(base_iters, 0.95):>16.1f}{pct(chunk_iters, 0.95):>18.1f}")
print(f"{'解码请求 TBT 最大 (ms)':<26}{max(base_iters):>16.1f}{max(chunk_iters):>18.1f}")
print(f"{'长请求 TTFT (ms)':<26}{base_ttft * 1000:>16.1f}{chunk_ttft * 1000:>18.1f}")
print(f"{'24 轮总耗时 (ms)':<26}{base_total * 1000:>16.1f}{chunk_total * 1000:>18.1f}")

# %% [markdown]
# ### 结果解读
#
# 大概率会看到这样一组数字：
#
# | 现象 | 说明 |
# |---|---|
# | **TBT 最大值大幅下降** | 整段 prefill 时有一轮迭代耗时是其他轮的好几倍；分块后每轮都均匀 |
# | **长请求 TTFT 变长** | 它要等好几轮才轮到结束，这是明确的代价 |
# | **总耗时基本不变** | **总计算量没变**，只是把阻塞摊平了 |
#
# 这就是 chunked prefill 的本质：**它不是优化，是重新分配**。用长请求自己的 TTFT，换所有其他请求的尾延迟。
#
# 值不值得看业务：长请求占比低（比如 5%）而 decode 请求海量时，这笔交易非常划算——5% 的用户多等一点，95% 的用户不再卡顿。

# %% [markdown]
# ## 四、什么时候不该用它
#
# 面试官喜欢追问边界条件，这三个都是真实的：
#
# 1. **chunk 切得太小**：每个 chunk 的矩阵乘太小，GPU 算力利用率骤降，吞吐反而变差。粒度要在"阻塞时间"和"算力效率"之间取平衡。
# 2. **显存压力大时**：分块 prefill 让更多请求同时处于"进行中"，KV cache 峰值占用上升，可能触发抢占。用显存换延迟，账要算清楚。
# 3. **本来就延迟不敏感**：离线批量推理只关心吞吐，chunked prefill 只带来额外调度开销。
#
# 还有一个容易忽略的：**它和 prefix caching 有重叠**。如果 prompt 大部分能命中缓存，prefill 本来就短，chunk 的意义就不大。两个优化不要重复投入。

# %% [markdown]
# ## 五、面试话术
#
# **问：chunked prefill 为什么能改善尾延迟？代价是什么？**
#
# - **问题**：prefill 是不可中断的一次大计算，长 prompt 独占 GPU，让同批正在解码的请求排队，表现为 TBT 尖刺。
# - **做法**：把 prefill 按 block 切块，每轮迭代算一块，与该轮 decode 混批。
# - **收益**：把一次长阻塞摊成多次短阻塞，TBT p99 显著下降，延迟分布变得可预测。
# - **代价**：长请求自身 TTFT 变长；chunk 过小降低算力效率；同时进行中的请求变多，KV 显存峰值上升。
# - **本质**：总计算量不变，是延迟在请求之间的**重新分配**——用少数长请求的 TTFT 换整体的尾延迟。
#
# 最后那句"不是优化而是重新分配"是这句话的分水岭。它说明你知道自己在做什么交易，而不是在执行一个听说过名字的技术。
#
# **作业**
#
# 1. 把 `chunk` 从 256 改成 64 和 1024，重跑 episode，观察 TBT 最大值 / TTFT / 总耗时三个指标怎么变，找出你机器上的平衡点。
# 2. 把 `n_decode` 从 8 改成 32，长 prefill 的阻塞效应是变强还是变弱？为什么？
# 3. 思考题：如果长请求的用户体验很重要（比如付费用户），设计什么机制既保住他的 TTFT 又保住其他人的尾延迟？（提示：优先级 + 抢占）
#
# **下一章**：换个方向提速——用一个小模型给大模型"打草稿"，也就是投机解码。

