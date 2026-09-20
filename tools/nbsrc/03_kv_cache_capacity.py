# %% [markdown]
# # 03 · KV Cache 与显存容量计算
#
# 这是**推理 infra 面试的必考题**，而且是最容易当场露馅的一题：
#
# > 一台 80G 的卡，跑 Llama-3-8B，4K 上下文，最多能同时服务多少个请求？
#
# 答不上来的人，说明从没真正做过容量规划。这一章把公式推清楚、用代码验证、然后做成一个随时能用的计算器。

# %%
# @@SETUP@@

# %%
# 本章的实验对象。第 01 章详细拆解过它的结构，这里直接建出来用。
model = build_model()
print(f"MiniGPT: {model.n_params / 1e6:.1f}M 参数")

# %% [markdown]
# ## 一、先建立代价直觉：KV cache 省了多少计算
#
# 不用 KV cache 时，每生成一个 token 都要把前面所有 token 重新算一遍。prompt 越长，浪费越大——而且不是线性浪费，是**平方级**的。

# %%
prompt_lens = [64, 256, 512]
N_OUT = 32

print(f"生成 {N_OUT} 个 token，对比两种实现的耗时\n")
print(f"{'prompt 长度':>12}{'无cache(ms)':>14}{'有cache(ms)':>14}{'加速比':>10}")
print("-" * 52)

for L in prompt_lens:
    p = torch.randint(0, model.cfg.vocab_size, (1, L), device=DEVICE)
    t_naive = bench(lambda: generate_naive(model, p, N_OUT), warmup=1, iters=3)
    t_cached = bench(lambda: generate_cached(model, p, N_OUT), warmup=1, iters=3)
    print(f"{L:>12}{t_naive:>14.1f}{t_cached:>14.1f}{t_naive / t_cached:>9.2f}x")

print()
print("prompt 越长，KV cache 的收益越大——因为没有它，每一步都要重算整段 prompt。")
print("但缓存不是免费的，代价就是显存。下面算它到底占多少。")

# %% [markdown]
# ## 二、公式推导
#
# 每一层、每个 token、每个 KV head 都要存一份 K 和一份 V：
#
# ```
# KV cache 字节数 = 2 × n_layer × n_kv_head × head_dim × seq_len × batch × dtype_bytes
#                    ↑   ↑          ↑              ↑
#                    |   |          |              └─ 每元素字节数（fp16 = 2）
#                    |   |          └─ 每个头的维度
#                    |   └─ KV head 的个数（GQA 下比 attention head 少）
#                    └─ K 和 V 各一份
# ```
#
# **两个高频陷阱**：
#
# 1. 忘记乘 2（K 和 V）。
# 2. 用 attention head 数计算。用了 GQA 的模型，KV head 数远小于 attention head 数，这是 GQA 存在的全部意义。

# %%
# 在 MiniGPT 上实测，验证公式
idx = torch.randint(0, model.cfg.vocab_size, (2, 128), device=DEVICE)
_, past = model(idx)

actual = sum(t.numel() * t.element_size() for layer in past for t in layer)
theory = kv_bytes(model.cfg.n_layer, model.cfg.n_head, model.cfg.head_dim, seq_len=128, batch=2,
                  dtype_bytes=2)

print(f"实测 past 张量总字节: {actual / 1024:8.1f} KB")
print(f"公式计算值          : {theory / 1024:8.1f} KB")
print(f"一致                : {actual == theory}")
print()
print("公式是对的。注意 past 里每层有两个张量 (k, v)，shape 都是 (batch, n_head, seq, head_dim)。")

# %%
# 每秒、每 token 的 KV cache —— 这是做容量规划时最常用的单位
def kv_per_token_kb(n_layer, n_kv_head, head_dim, dtype_bytes=2):
    return kv_bytes(n_layer, n_kv_head, head_dim, seq_len=1, dtype_bytes=dtype_bytes) / 1024


print(f"{'模型':<26}{'层数':>5}{'KV头':>6}{'每token':>12}{'4K上下文/请求':>16}")
print("-" * 68)
for name, cfg in [
    ("Llama-3-8B (GQA)", dict(n_layer=32, n_kv_head=8, head_dim=128)),
    ("Llama-3-8B (假如是 MHA)", dict(n_layer=32, n_kv_head=32, head_dim=128)),
    ("Qwen2.5-7B (GQA)", dict(n_layer=28, n_kv_head=4, head_dim=128)),
    ("Llama-3-70B (GQA)", dict(n_layer=80, n_kv_head=8, head_dim=128)),
]:
    per_tok = kv_per_token_kb(**cfg)
    four_k = per_tok * 4096 / 1024 / 1024  # KB → GiB
    print(f"{name:<26}{cfg['n_layer']:>5}{cfg['n_kv_head']:>6}{per_tok:>10.0f} KB{four_k:>14.2f} GB")

print()
print("两个值得记住的结论：")
print("  1. Llama-3-8B 每 token 约 128 KB，4K 上下文一个请求就要 0.5 GB。")
print("  2. 同样是 8B 模型，MHA 版本的 KV cache 是 GQA 的 4 倍——这就是为什么现在所有模型都用 GQA。")

# %% [markdown]
# ## 三、容量计算器
#
# 现在把公式反过来用：给定卡和模型，算出最大并发。

# %%
def capacity(gpu_mem_gb, params_b, n_layer, n_kv_head, head_dim, seq_len,
             weight_bytes=2, kv_elem_bytes=2, overhead_ratio=0.12):
    """返回 (最大并发, 总显存GB, 权重GB, 每请求KV GB)。

    overhead_ratio 预留激活值、CUDA graph、通信 buffer、显存碎片等开销。
    真实系统里 vLLM 还会把 block 内碎片算进去，这里取 12% 是个经验值。
    """
    total_bytes = gpu_mem_gb * (1 - overhead_ratio) * 1024 ** 3
    weight_bytes_total = params_b * 1e9 * weight_bytes
    avail = total_bytes - weight_bytes_total
    per_req = kv_bytes(n_layer, n_kv_head, head_dim, seq_len, batch=1, dtype_bytes=kv_elem_bytes)
    if avail <= 0:
        return 0, gpu_mem_gb, weight_bytes_total / 1024 ** 3, per_req / 1024 ** 3
    return int(avail // per_req), gpu_mem_gb, weight_bytes_total / 1024 ** 3, per_req / 1024 ** 3


print("场景：80G 卡（如 H100/A100-80G），fp16 权重\n")
print(f"{'模型':<22}{'上下文':>8}{'权重GB':>9}{'每请求KB':>11}{'最大并发':>10}")
print("-" * 62)
for name, p, cfg in [
    ("Llama-3-8B", 8.03, dict(n_layer=32, n_kv_head=8, head_dim=128)),
    ("Qwen2.5-7B", 7.62, dict(n_layer=28, n_kv_head=4, head_dim=128)),
    ("Llama-3-70B", 70.6, dict(n_layer=80, n_kv_head=8, head_dim=128)),
]:
    for seq in [2048, 8192]:
        conc, _, wgb, perkb = capacity(80, p, seq_len=seq, **cfg)
        print(f"{name:<22}{seq:>8}{wgb:>9.1f}{perkb * 1024 * 1024:>11.0f}{conc:>10}")

# %% [markdown]
# ### 把数字读一遍
#
# 几个可以直接用在面试里的结论：
#
# - **8B 模型在 80G 卡上，2K 上下文约 220 并发，4K 掉到 110，8K 只剩 55**。这就是为什么线上要拆多副本，而不是指望单卡扛住全部流量。
# - **70B 模型光权重就 130+ GB，单张 80G 卡连放都放不下**（上面表里会显示 0 并发）。必须先做张量并行切分或者量化到 FP8/INT4——这是"为什么需要模型并行"最直接的解释。
# - **上下文长度翻倍，并发直接减半**。这解释了一个常见的线上现象：QPS 没变，只是把 `max_model_len` 调大了，服务却开始排队。
#
# **想提高并发，按效果排序**：
#
# | 手段 | 效果 | 代价 |
# |---|---|---|
# | 换 GQA/MLA 模型或降低 KV 精度 | KV 减半到 1/4 | 精度损失，需要评估 |
# | 限制 `max_model_len` | 线性提升 | 长文档场景不可用 |
# | 权重量化到 FP8/INT4 | 权重减半到 1/4，腾出空间给 KV | 精度损失 |
# | 张量并行 TP | 每卡权重和 KV 都减半 | 通信开销，卡间互联要求高 |

# %% [markdown]
# ## 四、面试怎么答这道题
#
# 被问到"80G 卡跑 8B 模型最多多少并发"，按这个顺序说：
#
# 1. **先问清楚前提**：上下文长度多少？KV 用 fp16 还是 fp8？权重什么精度？有没有其他模型混布？
# 2. **给出公式**：`2 × 层数 × KV头数 × head_dim × 序列长度 × 并发 × 字节数`，强调**用 KV head 而不是 attention head**（这是最常见的错误）。
# 3. **算权重**：8B × 2 字节 = 16 GB，再留 10% 左右的激活和碎片开销。
# 4. **算可用空间**：80G × 0.88 - 16G ≈ 54 GB。
# 5. **算每请求 KV**：8K 上下文约 0.5 GB → 大约 100 条并发。
# 6. **补一句工程现实**：这是理论值，实际还要扣掉 block 内碎片（第 05 章）、prefill 峰值显存、以及调度上要留的余量，通常按 70-80% 打折规划。
#
# 第 6 步是拉开差距的地方——说明你做过真实部署，不是只会套公式。
#
# **作业**
#
# 1. 用 `capacity()` 算一下：如果 KV cache 量化到 fp8（`kv_elem_bytes=1`），并发能提升多少？
# 2. 如果要支撑 500 并发、8K 上下文跑 Llama-3-8B，需要几张 80G 卡？
# 3. 思考题：为什么 vLLM 的 PagedAttention 能把并发做得更高？（提示：和第 05 章的显存碎片有关）
#
# **下一章**：既然单个请求喂不饱 GPU，怎么把不同长度、不同进度的请求拼进同一个 batch——continuous batching。
