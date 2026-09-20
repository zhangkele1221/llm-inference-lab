# %% [markdown]
# # 01 · 环境与基准工具
#
# 这一章做三件事：确认 Colab 的 GPU 环境、建立后面九章都要用的测量工具、定义一个完全可控的 MiniGPT。
#
# **为什么先建测量工具？**
#
# 推理 infra 的一切结论都建立在数字上。"延迟降低 40%"这句话如果没交代测量口径，就是废话——是在 batch=1 还是 batch=64 下测的？测的是单次调用还是含队排队的端到端？有没有做 warmup？
#
# 面试官问"这个数字怎么来的"时，能讲清口径的人，和只会背数字的人，是两种候选人。这一章就是为后者准备的。
#
# **运行前检查**：Colab 菜单 → 代码执行程序 → 更改运行时类型 → 硬件加速器选 **GPU**。

# %%
import sys
import platform

import torch

print(f"Python    : {sys.version.split()[0]}")
print(f"平台      : {platform.platform()}")
print(f"PyTorch   : {torch.__version__}")
print(f"CUDA 可用 : {torch.cuda.is_available()}")

if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"GPU 型号  : {p.name}")
    print(f"显存      : {p.total_memory / 1024 ** 3:.1f} GB")
    print(f"计算能力  : sm_{p.major}{p.minor}")
    print(f"SM 数量   : {p.multi_processor_count}")
else:
    print()
    print("⚠️  没有检测到 GPU")
    print("   Colab：菜单 → 代码执行程序 → 更改运行时类型 → 硬件加速器 选 GPU")
    print("   第 01-08 章在 CPU 上也能跑完，只是慢；第 09、10 章必须要 GPU。")

# %%
!nvidia-smi

# %% [markdown]
# ## 一、引导单元：测量工具 + MiniGPT
#
# 下面这个单元格在每个 notebook 里都有一份完整副本，保证任何一章都能独立运行。
#
# 它包含五样东西：
#
# 1. **显卡规格 `CARD_SPECS` 和 `SPEC`**：你的卡有多少显存、多少带宽、多少算力。第 02、03 章直接拿它算账。
# 2. `sync / bench / peak_mem_mb`：测量工具。GPU 是**异步执行**的，不调 `torch.cuda.synchronize()` 就计时，测到的是 kernel 下发时间而不是执行时间——这是新手最常犯的测量错误。
# 3. `MiniGPT`：结构与 Llama 同源的因果语言模型，约 2700 万参数。
# 4. `generate_naive / generate_cached`：两条生成路径，用来对比有无 KV cache。
# 5. `kv_bytes`：KV cache 显存公式，第 03 章会实测验证它。
#
# **关于 MiniGPT 的一个重要设计**：它的 `forward` 接受 `pos_offset` 参数，允许 KV cache 从任意位置继续。这正是实现连续批处理的前提——不同请求处在不同位置，调度器必须能把它们拼进同一个 batch。

# %%
# @@SETUP@@

# %% [markdown]
# ## 二、你的卡是什么规格
#
# 后面每一章都要用这三个数字：
#
# | 参数 | 含义 | 为什么重要 |
# |---|---|---|
# | 显存容量 | 能装多少权重 + KV cache | 第 03 章算最大并发 |
# | 显存带宽 | 每秒能从显存搬多少字节 | **decode 阶段的瓶颈就在这里** |
# | FP16 算力 | 每秒能做多少次浮点运算 | **prefill 阶段的瓶颈** |
#
# 把「算力 ÷ 带宽」算出来，你就有了判断一个操作是 compute-bound 还是 memory-bound 的标尺。第 02 章会用到。
#
# > 表里是常见卡的近似规格（FP16 稠密算力，不含稀疏加速）。**你的卡不在表里的话，直接在上面那个引导单元里补一行**，三个数字在厂商 datasheet 上都能查到。

# %%
for k, v in SPEC.items():
    print(f"{k:12s}: {v}")

if SPEC["bw_gbps"]:
    ratio = SPEC["fp16_tflops"] * 1e12 / (SPEC["bw_gbps"] * 1e9)
    print(f"\n算力/带宽比 = {ratio:.0f} FLOP/byte")
    print("→ 记住这个数，第 02 章用它判断 prefill 和 decode 各自的瓶颈。")

# %% [markdown]
# ## 三、认识你的实验对象
#
# MiniGPT 的参数配置：4 层、6 个注意力头、隐藏维度 384、词表 50257、上下文 1024。
#
# 这个规模是刻意选的：它小到能在 T4 上秒级完成任务，又大到足以让显存带宽和调度开销呈现出和真实大模型相同的规律。**我们要观察的是比例关系，不是绝对值。**

# %%
model = build_model()

print(f"结构      : {model.cfg.n_layer} 层 / {model.cfg.n_head} 头 / head_dim={model.cfg.head_dim}")
print(f"参数量    : {model.n_params / 1e6:.1f} M")
print(f"权重显存  : {model.n_params * 2 / 1024 ** 2:.1f} MB (fp16)")

reset_peak()
_ = model(torch.randint(0, model.cfg.vocab_size, (1, 128), device=DEVICE))
print(f"跑一次 128 token 的峰值显存: {peak_mem_mb():.1f} MB")

# %% [markdown]
# ## 四、第一次测量：prefill 一次 vs 逐 token 生成
#
# 两种操作的计算形态完全不同：
#
# - **prefill**：一次性把整段 prompt 喂进去，是一大块稠密矩阵乘法。批量大、并行度高。
# - **decode**：一次只生成一个 token，每步都要把**全部权重**从显存读一遍。批量小、访存密集。
#
# 先建立直觉，第 02 章会定量分析。

# %%
B, T = 8, 512
idx = torch.randint(0, model.cfg.vocab_size, (B, T), device=DEVICE)

reset_peak()
ms = bench(lambda: model(idx), warmup=3, iters=10)
tokens = B * T
print(f"prefill {B} 条 × {T} token")
print(f"  耗时      : {ms:.1f} ms")
print(f"  吞吐      : {tokens / (ms / 1000):,.0f} token/s")
print(f"  峰值显存  : {peak_mem_mb():.0f} MB")

# %%
prompt = torch.randint(0, model.cfg.vocab_size, (1, 64), device=DEVICE)
N = 32

ms_naive = bench(lambda: generate_naive(model, prompt, N), warmup=1, iters=3)
ms_cached = bench(lambda: generate_cached(model, prompt, N), warmup=1, iters=3)

print(f"生成 {N} 个 token（prompt 长度 64）")
print(f"  不用 KV cache : {ms_naive:8.1f} ms")
print(f"  使用 KV cache : {ms_cached:8.1f} ms")
print(f"  加速比        : {ms_naive / ms_cached:.2f}x")

# %% [markdown]
# ### 顺手做一个正确性验证
#
# 性能优化最怕的是"变快了但算错了"。`generate_cached` 用 KV cache 只算新 token 的 Q/K/V，`generate_naive` 每步重算全部——**两者的输出必须逐位相同**。
#
# 这个习惯要带到第 05 章：验证前缀复用时，同样用"逐位比对 logits"而不是"看起来差不多"。

# %%
a = generate_naive(model, prompt, N)
b = generate_cached(model, prompt, N)
print("两条路径输出完全一致:", torch.equal(a, b))

# %% [markdown]
# ## 五、小结与作业
#
# **本章产出**：一个可复用的测量框架，一个可控的模型，以及一条正确的性能验证方法。
#
# 三个容易踩的坑，后面每一章都会反复遇到：
#
# 1. **忘记 `torch.cuda.synchronize()`** → 测出来是下发时间，数字好看得离谱。
# 2. **忘记 warmup** → 第一次调用包含 kernel 编译、显存分配器预热，数字难看。
# 3. **只看平均值不看分布** → 推理 infra 里 p99 往往比均值更重要，第 06 章会专门处理尾延迟。
#
# **作业（做完再进第 02 章）**
#
# 1. 把 `CARD_SPECS` 里你的卡补全，手算「算力 ÷ 带宽」得到算力带宽比。
# 2. 把 `bench()` 的 `iters` 从 10 改成 100，观察数字是否稳定；如果不稳定，想想是什么在干扰（提示：Colab 是共享实例）。
# 3. 把 MiniGPT 改成 `n_layer=8`，重复上面的测量，看看参数量翻倍对 prefill 和 decode 的影响是否一样。
#
# **下一章**：用刚才记下的算力带宽比，定量分析 prefill 和 decode 各自的瓶颈，并实测验证。
