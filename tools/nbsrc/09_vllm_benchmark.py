# %% [markdown]
# # 09 · vLLM 实战：压测与指标解读
#
# 前八章都是手写实现，目的是让你**知道内部发生了什么**。这一章切回真实生产框架，验证两件事：
#
# 1. 你手写过的那些机制，在 vLLM 里对应什么、怎么观测。
# 2. **prompt 结构如何影响 prefix cache 命中率**——这是能直接换算成钱的优化。
#
# > 这一章需要 GPU。Colab 菜单 → 代码执行程序 → 更改运行时类型 → 硬件加速器选 GPU。
# %%
# 安装 vLLM（约 2GB，需要几分钟）。装完可能提示重启运行时，按提示操作即可。
!pip install -q vllm
# %%
import time

import torch

print(f"CUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"GPU: {p.name}  {p.total_memory / 1024 ** 3:.1f} GB  sm_{p.major}{p.minor}")

import vllm
print(f"vLLM 版本: {vllm.__version__}")
# %% [markdown]
# ## 一、加载模型
#
# 用 Qwen2.5-0.5B-Instruct：够小，T4 上跑得舒服；又足够真实，走完整的 vLLM 推理路径。
#
# 注意两个参数：
#
# - `enable_prefix_caching=True`：打开前缀缓存，本章核心实验依赖它。
# - `max_model_len`：**直接决定 KV cache 占用和并发上限**，第 03 章算过。线上调大它之前先算账。
# %%
from vllm import LLM, SamplingParams

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

llm = LLM(
    model=MODEL,
    enable_prefix_caching=True,
    max_model_len=4096,
    gpu_memory_utilization=0.85,
    dtype="float16",
)

sampling = SamplingParams(temperature=0.0, max_tokens=64)
print("模型加载完成")
# %% [markdown]
# ## 二、基线吞吐
#
# vLLM 的 offline 接口一次吃一批 prompt，内部自动组 continuous batch。先建立基线。
# %%
def run_batch(prompts, sampling_params=None):
    sampling_params = sampling_params or sampling
    t0 = time.perf_counter()
    outs = llm.generate(prompts, sampling_params)
    dt = time.perf_counter() - t0
    n_out = sum(len(o.outputs[0].token_ids) for o in outs)
    n_in = sum(len(o.prompt_token_ids) for o in outs)
    return outs, dt, n_in, n_out


base_prompts = [f"用三句话解释什么是显存带宽。这是第 {i} 个问题。" for i in range(32)]

outs, dt, n_in, n_out = run_batch(base_prompts)
print(f"32 条请求，输入 {n_in} token，输出 {n_out} token")
print(f"总耗时   : {dt:.2f} s")
print(f"输出吞吐 : {n_out / dt:,.0f} token/s")
print(f"请求吞吐 : {len(base_prompts) / dt:,.1f} req/s")
# %% [markdown]
# ## 三、核心实验：prompt 结构决定 prefix cache 命中率
#
# 回想第 05 章的结论：**链式哈希要求从第一个 block 起连续匹配**。现在用真实框架验证。
#
# 设计两组请求，**总 token 数完全相同**，唯一区别是变量放在哪：
#
# - **A 组**：长公共模板在前，变量在后 → 前缀可缓存
# - **B 组**：变量在最前面，长公共模板在后 → 第一个 block 就不同，缓存全失效
# %%
LONG_TEMPLATE = (
    "你是一个严谨的技术助手。请遵守以下回答规范："
    "第一，先给出结论再展开理由。"
    "第二，涉及性能数据时必须说明测量口径，包括硬件型号、批次大小、并发数。"
    "第三，涉及方案对比时必须同时给出代价，不能只讲收益。"
    "第四，回答控制在三句话以内。"
) * 6      # 重复放大，让前缀足够长，效果更明显

questions = [f"问题编号 {i}：什么是 KV cache？" for i in range(32)]
prompts_a = [LONG_TEMPLATE + "\n" + q for q in questions]   # 模板在前
prompts_b = [q + "\n" + LONG_TEMPLATE for q in questions]   # 变量在前

tok = llm.get_tokenizer()
print(f"A 组单条长度: {len(tok.encode(prompts_a[0]))} token")
print(f"B 组单条长度: {len(tok.encode(prompts_b[0]))} token")
print("两组长度相同，只有变量位置不同。")

# 预热，让缓存状态稳定
_ = llm.generate([prompts_a[0]], sampling)
_ = llm.generate([prompts_b[0]], sampling)
# %%
outs_a, dt_a, n_in_a, n_out_a = run_batch(prompts_a)
outs_b, dt_b, n_in_b, n_out_b = run_batch(prompts_b)

print(f"{'':<16}{'输入token':>12}{'耗时(s)':>12}{'相对速度':>12}")
print("-" * 52)
print(f"{'A 模板在前':<16}{n_in_a:>12}{dt_a:>12.2f}{1.0:>11.2f}x")
print(f"{'B 变量在前':<16}{n_in_b:>12}{dt_b:>12.2f}{dt_b / dt_a:>11.2f}x")
print()
print(f"同样的输入规模，A 组比 B 组快 {dt_b / dt_a:.2f} 倍。")
print()
print("差异全部来自 prefix cache：A 组 32 条请求共享同一段模板前缀，第一条之后")
print("只需计算各自的短后缀；B 组第一条就不同，32 条全部要完整 prefill。")
# %% [markdown]
# ### 这个实验的工程价值
#
# 你线上可能已经在跑同样的模板，只是没人检查过字段顺序。**把变量从 prompt 开头挪到结尾，是一行代码的改动，却可能带来 30% 以上的成本下降。**
#
# 上线前值得做一次 prompt 审计，把每个字段按位置列出来，问三个问题：
#
# 1. 这个字段每个请求都一样吗？一样就往前提。
# 2. 这个字段是变量吗？是就往后放。
# 3. 有没有请求 ID、时间戳、随机种子这类"每次都不同"的东西混在最前面？这最致命。
# %% [markdown]
# ## 四、并发对吞吐的影响
#
# 回到第 02 章的结论：单请求喂不饱 GPU，要靠 batching。用真实框架验证一遍。
# %%
print(f"{'批大小':>8}{'输出吞吐(t/s)':>16}{'平均每请求耗时(s)':>20}")
print("-" * 46)
for n in [1, 4, 16, 64]:
    prompts = base_prompts[:n] if n <= 32 else (base_prompts * 2)[:n]
    outs, dt, _, n_out = run_batch(prompts)
    print(f"{n:>8}{n_out / dt:>16,.0f}{dt / n:>19.3f}")

print()
print("批大小上升 → 总吞吐大幅提升，但**单请求耗时也变长**（要等其他人）。")
print("这是推理服务的根本矛盾：吞吐和延迟不可兼得，只能靠调度策略决定把资源分给谁。")
print("所以线上一定要分池：交互式请求走小批低延迟池，离线任务走大批高吞吐池。")
# %% [markdown]
# ## 五、怎么读 vLLM 的线上指标
#
# offline 接口看不到内部状态。真上生产要用 server 模式，vLLM 会暴露 Prometheus 指标（`/metrics`）。这几个最该盯：
# %%
KEY_METRICS = [
    ("vllm:num_requests_running", "正在跑的请求数", "持续贴着上限说明容量到顶"),
    ("vllm:num_requests_waiting", "排队中的请求数", "大于 0 就是容量不足的明确信号"),
    ("vllm:gpu_cache_usage_perc", "KV cache 占用率", "长期 >90% 快触发抢占；长期偏低说明显存配多了"),
    ("vllm:gpu_prefix_cache_hit_rate", "前缀缓存命中率", "低于预期就去审计 prompt 字段顺序"),
    ("vllm:time_to_first_token_seconds", "TTFT 分布", "用户感知的响应速度，看 p99"),
    ("vllm:time_per_output_token_seconds", "TPOT 分布", "打字流畅度，看 p99"),
    ("vllm:request_prefill_time_seconds", "prefill 耗时", "和 decode 分开看才能判断瓶颈阶段"),
    ("vllm:request_decode_time_seconds", "decode 耗时", "同上"),
]

print(f"{'指标':<40}{'含义':<24}怎么用")
print("-" * 108)
for name, meaning, how in KEY_METRICS:
    print(f"{name:<40}{meaning:<24}{how}")

print()
print("指标名在不同版本间会变，用之前先访问 /metrics 确认一遍。")
# %% [markdown]
# ## 六、现场定位：吞吐只有预期的一半，怎么查
#
# 这是推理 infra 面试里最像真实工作的题。给你一套可以照着走的方法论：
# %%
DIAGNOSIS = """
第一步 · 先分清瓶颈在哪个阶段（不要急着调参数）
    看 prefill 耗时 vs decode 耗时占总时间的比例。
    prefill 占比高 → 问题在输入侧：prompt 太长、缓存没命中、chunk 没开。
    decode 占比高  → 问题在输出侧：batch 没打满、KV 快满了、投机解码没开。

第二步 · 看排队
    num_requests_waiting > 0 说明容量不足。
    但如果 waiting 为 0 而吞吐仍不达标，说明是单请求效率问题，不是容量问题——
    这两个方向的解法完全相反，必须先分清。

第三步 · 看 KV cache 占用
    长期贴顶 → 并发上不去：量化 KV、降 max_model_len，或者加副本。
    长期很低 → 显存配多了，或者请求根本没并发起来。

第四步 · 看 prefix cache 命中率
    低于预期 → 审计 prompt 字段顺序（本章实验就是模板）。

第五步 · 才轮到调参数
    max_num_seqs / max_num_batched_tokens / gpu_memory_utilization /
    enable_chunked_prefill。参数是最后一步，不是第一步。

第六步 · 确认输入分布
    线上流量的输入长度是什么分布？p99 有多长？
    很多人不知道自己的 p99 输入长度，这是排查不下去的常见原因。
"""
print(DIAGNOSIS)

print("这套顺序的价值在于：**先分类，再动手**。")
print("新手最常见的错误是一上来就调 batch size——如果瓶颈其实在缓存没命中，调参数是白费。")
# %% [markdown]
# ## 七、面试话术与作业
#
# **问：你用过 vLLM 吗？**
#
# 别答"用过"。按这个结构答：
#
# 1. **先说场景**：什么业务、什么模型、QPS 和延迟要求是多少。
# 2. **再说你改了什么**：如果只是改了启动参数，老实说；如果读过 Scheduler、BlockManager，或者自己做过前缀路由，重点讲那个。
# 3. **然后给数字**：TTFT / TPOT / 吞吐 / 缓存命中率 / 每百万 token 成本，至少给两个。
# 4. **最后讲一个你踩过的坑**：比如 prompt 字段顺序导致缓存全失效、max_model_len 调大后开始排队。
#
# 第 4 条最能体现实战——踩过坑的人才知道坑在哪。
#
# **作业**
#
# 1. 把 `LONG_TEMPLATE` 再放大 3 倍，重跑 A/B 实验。倍数关系还在吗？
# 2. 给 B 组设计一个改造方案，在不改变语义的前提下让它也能命中缓存。
# 3. 把你线上 prompt（脱敏后）拿出来做一次字段位置审计，记录哪些字段会导致整段缓存失效。
#
# **下一章**：毕业项目——多副本时，怎么让相同前缀的请求落到同一个副本上。
