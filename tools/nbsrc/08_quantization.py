# %% [markdown]
# # 08 · 量化：权重与 KV Cache
#
# 量化是推理 infra 最直接的降本手段，也是面试里最容易被问细节的地方：
#
# > 你把模型量化到 INT8 了，那 KV cache 量化了吗？为什么 KV 量化比权重量化更敏感？
#
# 这一章手写量化算子，把显存收益和精度代价都量出来。

# %%
# @@SETUP@@

# %% [markdown]
# ## 一、对称线性量化
#
# 把所有浮点数映射到一个整数网格上：
#
# ```
# scale = max(|x|) / qmax
# q     = round(x / scale)     再截断到 [-qmax, qmax]
# x'    = q × scale            ← 反量化，这一步必然有损
# ```
#
# INT8 的 `qmax = 127`，INT4 的 `qmax = 7`。位数越少，网格越粗，误差越大。

# %%
def quantize_symmetric(x, n_bits=8, per_channel=False):
    """对称线性量化。per_channel=True 时每行独立算 scale，精度更好但要多存一份 scale。"""
    qmax = 2 ** (n_bits - 1) - 1
    if per_channel:
        dim = tuple(range(1, x.dim()))
        scale = x.abs().amax(dim=dim, keepdim=True) / qmax
    else:
        scale = x.abs().max() / qmax
    scale = scale.clamp(min=1e-8)
    q = torch.round(x / scale).clamp(-qmax, qmax)
    return q * scale, scale


x = torch.randn(512, 512, device=DEVICE)

print(f"{'位数':>6}{'粒度':>12}{'相对误差':>12}{'额外存储(scale)':>18}")
print("-" * 50)
for bits in [8, 4, 3]:
    for pc in [False, True]:
        xq, sc = quantize_symmetric(x, bits, pc)
        err = ((x - xq).abs().mean() / x.abs().mean()).item()
        extra = sc.numel() if pc else 1
        print(f"{bits:>6}{'per-channel' if pc else 'per-tensor':>12}{err:>11.1%}{extra:>16} 个")

print()
print("两个规律：")
print("  · 位数下降，误差迅速上升：8 位几乎无损，4 位开始明显，3 位基本不能用。")
print("  · per-channel 比 per-tensor 精度好，代价是要多存一组 scale——这是典型的")
print("    '用一点点存储换精度'的取舍。实际框架里大多是 per-channel 或分组量化。")

# %% [markdown]
# ## 二、权重量化：显存收益是确定的，精度代价要评估

# %%
model = build_model()

print(f"权重显存需求（{model.n_params / 1e6:.1f}M 参数）：\n")
print(f"{'精度':>10}{'每参数字节':>12}{'权重显存':>14}{'相对 FP16':>12}")
print("-" * 50)
fp16_mb = model.n_params * 2 / 1024 ** 2
for name, b in [("FP16", 2), ("INT8", 1), ("INT4", 0.5), ("FP8", 1)]:
    mb = model.n_params * b / 1024 ** 2
    print(f"{name:>10}{b:>12}{mb:>11.1f}MB{mb / fp16_mb:>11.0%}")

print()
print("权重减半、甚至压到四分之一，直接换来两个好处：")
print("  1. 显存腾出来给 KV cache → 并发提升（接第 03 章的容量计算器）")
print("  2. decode 每步要读的字节变少 → 速度直接提升")
print()
print("第 2 条特别值得注意：decode 是访存受限的，权重变小等于每步搬运量变小，")
print("所以量化对 decode 是'既省显存又快'，这是它比别的优化手段更受欢迎的原因。")

# %% [markdown]
# ## 三、KV Cache 量化：为什么更敏感
#
# KV cache 量化省显存的效果和权重一样（也是减半 / 压到 1/4），但**精度上危险得多**。原因是三个放大效应：
#
# 1. **它进的是 attention 的打分**。K 被量化后，Q·K 的相似度排序可能改变——而排序决定了模型"看哪里"，微小的数值误差会改变注意力分配的**结构**，不只是数值。
# 2. **误差会沿时间累积**。权重的误差是固定的，KV 的误差会随着序列变长、被反复读取而不断影响后续每一个 token。
# 3. **不同层的分布差异极大**。某些层的 K 有极端的离群值，per-tensor 的 scale 会被离群值拉大，导致其余数值全部挤在很粗的网格上。
#
# 实测一下第 2 条：

# %%
model = build_model(block_size=4096)
CTX = 256
ctx = torch.randint(0, model.cfg.vocab_size, (1, CTX), device=DEVICE)
logits_ref, past_ref = model(ctx)
nxt = logits_ref[:, -1].argmax(-1, keepdim=True)

logits_clean, _ = model(nxt, past_kvs=past_ref, pos_offset=CTX)

print(f"{'KV 精度':>10}{'logits 最大偏差':>18}{'top-1 是否一致':>16}")
print("-" * 46)
for bits, label in [(8, "INT8"), (4, "INT4"), (3, "INT3")]:
    past_q = []
    for k, v in past_ref:
        kq, _ = quantize_symmetric(k, bits, per_channel=True)
        vq, _ = quantize_symmetric(v, bits, per_channel=True)
        past_q.append((kq, vq))
    logits_q, _ = model(nxt, past_kvs=past_q, pos_offset=CTX)
    diff = (logits_clean[:, -1] - logits_q[:, -1]).abs().max().item()
    same = torch.equal(logits_clean[:, -1].argmax(-1), logits_q[:, -1].argmax(-1))
    print(f"{label:>10}{diff:>18.3f}{str(same):>16}")

print()
print("注意：这里只是**一步** decode 的偏差。真实生成要连续走几百步，每步都在")
print("被污染的 KV 上做注意力，误差会持续放大。所以 KV 量化的验收标准从来不是")
print("'单步 logits 差多少'，而是'整条链路的业务指标掉了多少'。")

# %% [markdown]
# ### 顺带算一下 KV 量化能换多少并发

# %%
def capacity(gpu_mem_gb, params_b, n_layer, n_kv_head, head_dim, seq_len,
             weight_bytes=2, kv_elem_bytes=2, overhead_ratio=0.12):
    total = gpu_mem_gb * (1 - overhead_ratio) * 1024 ** 3
    weight = params_b * 1e9 * weight_bytes
    avail = total - weight
    per_req = kv_bytes(n_layer, n_kv_head, head_dim, seq_len, dtype_bytes=kv_elem_bytes)
    return int(max(0, avail) // per_req)


print("Llama-3-8B，80G 卡，8K 上下文：\n")
print(f"{'配置':<34}{'最大并发':>10}{'相对 FP16':>12}")
print("-" * 56)
base = capacity(80, 8.03, 32, 8, 128, 8192, weight_bytes=2, kv_elem_bytes=2)
for label, wb, kb in [
    ("FP16 权重 + FP16 KV", 2, 2),
    ("FP16 权重 + INT8 KV", 2, 1),
    ("INT8 权重 + INT8 KV", 1, 1),
    ("INT4 权重 + INT8 KV", 0.5, 1),
]:
    c = capacity(80, 8.03, 32, 8, 128, 8192, weight_bytes=wb, kv_elem_bytes=kb)
    print(f"{label:<34}{c:>10}{c / base:>11.1f}x")

print()
print("这就是量化被普遍采用的原因：**并发能翻好几倍**，而代价只是精度需要评估。")

# %% [markdown]
# ## 四、几个必须知道的名词
#
# | 名词 | 是什么 | 什么时候用 |
# |---|---|---|
# | **W8A8** | 权重 8 位、激活 8 位 | 通用，硬件支持广 |
# | **W4A16** | 权重 4 位、激活仍是 16 位 | 显存紧张但算力充足时最常用 |
# | **AWQ** | 激活感知的权重量化，按重要性保护关键通道 | 4 位部署的主流方案之一 |
# | **GPTQ** | 逐层最小化重构误差的后训练量化 | 和 AWQ 并列的主流方案 |
# | **FP8 (E4M3/E5M2)** | 浮点格式而非整数，动态范围好 | H 系列及更新的卡才有硬件支持 |
# | **KV cache 量化** | 只压 KV，不压权重 | 长上下文场景收益最大 |
#
# **一个容易说错的点**：FP8 是**浮点**不是整数。它的指数位给了很好的动态范围，所以对付离群值比 INT8 更从容，不需要复杂的 per-channel scale。这也是新卡上 FP8 更受青睐的原因。
#
# **另一个常被追问的点**：为什么量化之后有些模型"变笨"了但困惑度几乎没变？因为困惑度是平均意义上的指标，而量化误差往往集中在少数关键 token 上——平均值看不出来，但生成质量会掉。所以**量化的验收一定要看业务指标，不能只看困惑度**。

# %% [markdown]
# ## 五、面试话术
#
# **问：量化的收益和代价？**
#
# - **收益有两份**：显存（腾给 KV cache，直接提升并发）和速度（decode 是访存受限，权重变小等于每步搬运量变小）。
# - **代价是精度**，而且必须用业务指标验收，不能只看困惑度。
#
# **问：为什么 KV cache 量化比权重量化更敏感？**
#
# - K 参与 attention 打分，量化误差会**改变注意力的分配结构**，不只是数值偏移。
# - 误差随时间**累积**：权重误差是固定的，KV 误差会被后续每一个 token 反复读取。
# - 各层的 K/V 分布差异大，离群值会把 per-tensor 的 scale 拉大，让其余数值的精度崩塌。
#
# **问：怎么选量化方案？**
#
# 按顺序排除：硬件支持什么格式（FP8 需要新卡）→ 显存缺多少（决定权重压到几位）→ 上下文多长（长上下文优先量化 KV）→ 精度能不能通过业务验收。先算账再选方案，而不是反过来。
#
# **作业**
#
# 1. 把 `quantize_symmetric` 改成**分组量化**（比如每 128 个元素一组算 scale），看看误差能改善多少。这就是真实框架里 GPTQ/AWQ 的做法基础。
# 2. 用 INT4 KV 连续 decode 32 步，统计有多少步的 top-1 token 和 FP16 版本不一致。
# 3. 思考题：为什么说"量化是唯一同时改善显存和延迟的优化手段"？其他手段（比如加并行、加批次）为什么做不到？
#
# **下一章**：从手写实现切到真实生产框架，用 vLLM 做压测并读它的线上指标。

