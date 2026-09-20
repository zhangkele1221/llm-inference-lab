# %% [markdown]
# # 02 · Prefill vs Decode：两个阶段，两种瓶颈
#
# 这一章回答一个面试高频问题：
#
# > **为什么说 prefill 是算力密集、decode 是显存带宽密集？**
#
# 背结论没用，面试官会追问"那 batch=1 的 decode 能跑多快"。这一章把公式推一遍，再实测验证。

# %%
# @@SETUP@@

# %%
# 本章的实验对象。第 01 章详细拆解过它的结构，这里直接建出来用。
model = build_model()
print(f"MiniGPT: {model.n_params / 1e6:.1f}M 参数 | 你的卡: {SPEC['name']}")

# %% [markdown]
# ## 一、理论：一个 token 要搬多少字节，做多少次浮点运算
#
# **前向传播的 FLOPs**：每个参数参与一次乘法加一次加法，所以处理 `T` 个 token 大约是
#
# ```
# FLOPs ≈ 2 × N_params × T
# ```
#
# **需要搬运的字节数**：权重必须从显存读进计算单元，
#
# ```
# Bytes ≈ N_params × dtype_bytes
# ```
#
# 两者相除就是**算术强度**（arithmetic intensity），单位是 FLOP/byte：
#
# | 阶段 | token 数 T | 算术强度 | 归属 |
# |---|---|---|---|
# | Prefill | T = prompt 长度 | `2NT / 2N` = **T** | T=512 时是 512 → 远高于卡的比值 → **compute-bound** |
# | Decode | T = 1 | `2N / 2N` = **1** | 远低于卡的比值 → **memory-bound** |
#
# 判断标准：**算术强度 > 卡的算力带宽比 → compute-bound，反之 memory-bound**。

# %%
def card_ratio(spec):
    """卡的算力带宽比（FLOP/byte）。"""
    if not spec["bw_gbps"]:
        return float("nan")
    return spec["fp16_tflops"] * 1e12 / (spec["bw_gbps"] * 1e9)


def forward_flops(n_params, n_tokens):
    return 2 * n_params * n_tokens


def decode_ceiling(n_params, bw_gbps, dtype_bytes=2, batch=1):
    """decode 的理论 token 速率上限。

    每生成一步要读一遍全部权重（N × dtype_bytes 字节），带宽决定了每秒能走多少步。
    batch 个请求共享同一次权重读取，所以吞吐随 batch 线性增长（直到算力或 KV 带宽成为新瓶颈）。
    """
    bytes_per_step = n_params * dtype_bytes
    return bw_gbps * 1e9 * batch / bytes_per_step


ratio = card_ratio(SPEC)
print(f"你的卡：{SPEC['name']}")
print(f"  算力带宽比 = {ratio:.0f} FLOP/byte\n")
print("各阶段算术强度 vs 卡片比值：")
for label, ai in [("prefill T=128", 128), ("prefill T=1024", 1024), ("decode batch=1", 1)]:
    verdict = "compute-bound" if ai > ratio else "memory-bound"
    print(f"  {label:16s} AI={ai:>6}  → {verdict}")

# %% [markdown]
# ## 二、先算一个真实模型的 decode 天花板
#
# 公式摆出来就要拿真模型验证。Llama-3-8B 在 T4 上，batch=1 每秒最多能吐多少 token？

# %%
def human(n):
    for unit in ["", "K", "M", "B"]:
        if abs(n) < 1000:
            return f"{n:.1f}{unit}"
        n /= 1000
    return f"{n:.1f}T"


print(f"{'模型':<18}{'参数量':>10}{'权重占用':>12}{'decode 上限(batch=1)':>22}")
print("-" * 64)
for name, params in [("MiniGPT", model.n_params), ("Llama-3-8B", 8.03e9), ("Qwen2.5-7B", 7.62e9)]:
    ceiling = decode_ceiling(params, SPEC["bw_gbps"])
    print(f"{name:<18}{human(params):>10}{params * 2 / 1024 ** 3:>10.1f}GB{ceiling:>20,.0f} t/s")

print()
print("这张表值得记住：")
print("  大模型在单卡上的 decode 速度，被显存带宽死死卡住。")
print("  想更快只有三条路：换带宽更大的卡、量化把权重变小、或者一次喂多个请求摊薄权重读取。")

# %% [markdown]
# ## 三、实测 prefill：算力利用率能到多少
#
# 让序列长度从 128 涨到 1024，看达成的 TFLOPS 是否随序列变长而提升。

# %%
print(f"{'batch':>6}{'seq':>7}{'耗时(ms)':>12}{'吞吐(t/s)':>14}{'达成TFLOPS':>14}{'MFU':>9}")
print("-" * 64)

for B in [1, 4, 16]:
    for T in [128, 512, 1024]:
        idx = torch.randint(0, model.cfg.vocab_size, (B, T), device=DEVICE)
        ms = bench(lambda: model(idx), warmup=3, iters=10)
        tps = B * T / (ms / 1000)
        tflops = forward_flops(model.n_params, B * T) / (ms / 1000) / 1e12
        mfu = tflops / SPEC["fp16_tflops"] * 100 if SPEC["fp16_tflops"] else float("nan")
        print(f"{B:>6}{T:>7}{ms:>12.1f}{tps:>14,.0f}{tflops:>14.2f}{mfu:>8.2f}%")

print()
print("观察两点：")
print("  1. MFU 大概率很低（个位数百分点）。MiniGPT 太小，GPU 还没热身就结束了——")
print("     这说明 MFU 这个指标对大模型才有意义，小模型上瓶颈是 kernel 启动开销。")
print("  2. 序列越长、batch 越大，效率通常越好，因为矩阵乘法的并行度更高。")

# %% [markdown]
# ## 四、实测 decode：为什么必须做 batching
#
# 这是本章最重要的一段实验。固定上下文长度，只改 batch，看吞吐怎么变。

# %%
@torch.no_grad()
def decode_bench(model, batch, ctx_len, steps=32):
    """返回 (token/s, 每步毫秒)。prefill 不计入计时。"""
    idx = torch.randint(0, model.cfg.vocab_size, (batch, ctx_len), device=DEVICE)
    logits, past = model(idx)
    nxt = logits[:, -1].argmax(-1, keepdim=True)
    pos = ctx_len

    sync()
    t0 = time.perf_counter()
    for _ in range(steps):
        logits, past = model(nxt, past_kvs=past, pos_offset=pos)
        pos += 1
        nxt = logits[:, -1].argmax(-1, keepdim=True)
    sync()
    dt = (time.perf_counter() - t0) / steps
    return batch / dt, dt * 1000


CTX = 128
ceiling1 = decode_ceiling(model.n_params, SPEC["bw_gbps"], batch=1)

print(f"上下文长度 {CTX}，连续 decode 32 步")
print(f"理论带宽上限(batch=1) = {ceiling1:,.0f} token/s\n")
print(f"{'batch':>6}{'吞吐(t/s)':>14}{'每步(ms)':>12}{'占带宽上限':>14}")
print("-" * 48)

for B in [1, 4, 16, 64]:
    tps, ms_step = decode_bench(model, B, CTX)
    pct = tps / (ceiling1 * B) * 100
    print(f"{B:>6}{tps:>14,.0f}{ms_step:>12.2f}{pct:>13.1f}%")

print()
print("这里会出现一个反直觉的现象：")
print("  batch=1 时，达成率可能只有百分之几——GPU 大部分时间在等 kernel 下发，而不是在搬数据。")
print("  随着 batch 增大，同一份权重被更多请求摊薄，达成率快速逼近上限。")
print()
print("这就是 continuous batching 存在的全部理由：")
print("  单请求的 decode 根本无法喂饱 GPU，必须把多个请求拼在一起跑。")

# %% [markdown]
# ## 五、把结论整理成面试话术
#
# 问你"prefill 和 decode 有什么区别"，按这个结构答：
#
# 1. **计算形态不同**：prefill 一次处理整段 prompt，是大的稠密 GEMM，算术强度等于序列长度；decode 一次一个 token，算术强度约等于 1。
# 2. **瓶颈不同**：前者受算力限制（看 MFU 利用率），后者受显存带宽限制（看带宽利用率）。判断依据是算术强度和卡片算力带宽比的比较。
# 3. **工程含义不同**：因为 decode 是访存密集且单请求喂不饱 GPU，所以要 continuous batching；因为 prefill 吃算力且会长时间占住 GPU，长 prefill 会阻塞其他请求，所以要 chunked prefill（第 06 章）。
# 4. **可验证**：给他一个具体数字——比如 Llama-3-8B 在 T4 上 batch=1 的理论 decode 上限约为 20 token/s。
#
# **作业**
#
# 1. 把 `CTX` 从 128 改到 1024，重跑 decode 实验。随着上下文变长，KV cache 的读取量变大，达成率会怎么变？
# 2. 用 `decode_ceiling` 算一下：如果换成 H100（3350 GB/s），Llama-3-8B 的 batch=1 上限是多少？和 T4 差几倍？
# 3. 思考题：为什么增大 batch 能提升吞吐，但**不能降低单请求的延迟**？这对服务设计意味着什么？
#
# **下一章**：decode 阶段的 KV cache 到底占多少显存，一个 80G 的卡能同时服务多少请求。
