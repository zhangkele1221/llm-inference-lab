# %% [markdown]
# # 11 · vLLM 源码导读
#
# 前十章你手写了一遍 vLLM 的核心机制。这一章教你怎么**真正打开它的源码**，并且知道该看什么、看到什么程度就够了。
#
# **前置**：至少读完第 04、05 章。没有那两章打底，直接读源码会迷路——因为你会分不清哪些是核心机制、哪些是工程细节。
#
# > 这一章不需要 GPU，也不需要 MiniGPT。它是一份导读。
# %%
try:
    import vllm
    print(f"vLLM 已安装: {vllm.__version__}")
except ImportError:
    print("没装 vLLM。在 Colab 里先跑一次：")
    print("    !pip install -q vllm        # 约 2GB，需要几分钟")
    print()
    print("装完之后重启运行时，再回来继续这一章。")
# %% [markdown]
# ## 一、先在你自己的环境里定位源码
#
# 不要在网上找源码截图。**直接在你装的这个版本里读**——版本之间路径和类名会变，网上抄来的会误导你。
# %%
import pathlib

import vllm

ROOT = pathlib.Path(vllm.__file__).parent
print(f"vLLM 源码根目录: {ROOT}\n")

KEY_FILES = [
    ("v1/engine/core.py", "EngineCore：主循环"),
    ("v1/core/sched/scheduler.py", "Scheduler：决定跑什么"),
    ("v1/core/sched/output.py", "SchedulerOutput：调度与执行的接口"),
    ("v1/core/kv_cache_manager.py", "KVCacheManager：KV 放哪、能复用多少"),
    ("v1/core/block_pool.py", "BlockPool：物理 block 的分配与回收"),
    ("v1/core/kv_cache_utils.py", "KVCacheBlock / hash_block_tokens"),
    ("v1/request.py", "Request：num_computed_tokens 就在这"),
    ("v1/worker/gpu_model_runner.py", "GPUModelRunner：真正跑模型"),
    ("v1/worker/gpu_input_batch.py", "InputBatch：拼 batch 的账本"),
    ("config.py", "SchedulerConfig / CacheConfig：那些参数"),
]

print(f"{'文件':<42}{'作用':<30}存在")
print("-" * 78)
for rel, desc in KEY_FILES:
    p = ROOT / rel
    flag = "✓" if p.exists() else "✗ 版本可能变了，用 rg 找"
    print(f"{rel:<42}{desc:<30}{flag}")
# %% [markdown]
# ## 二、调用链全景
#
# vLLM V1 的一次推理迭代，从入口到底层是这样的：
# %%
CALL_CHAIN = """
EngineCore.step()                                 vllm/v1/engine/core.py
  │
  ├─ 1. 调度
  │    Scheduler.schedule()                       vllm/v1/core/sched/scheduler.py
  │      ├─ 先排 running 里的请求（decode 各 1 个 token，chunked prefill 按剩余量）
  │      ├─ 再排 waiting 里的新请求（prefill）
  │      ├─ 每轮受 max_num_batched_tokens 预算约束   ← chunked prefill 就出在这
  │      ├─ 为每个请求向 KVCacheManager 申请 block
  │      │    └─ KVCacheManager.allocate_slots()  vllm/v1/core/kv_cache_manager.py
  │      │         └─ BlockPool.get_new_blocks()  vllm/v1/core/block_pool.py
  │      └─ 产出 SchedulerOutput                  vllm/v1/core/sched/output.py
  │
  ├─ 2. 执行
  │    Executor.execute_model()                   多进程 / 单进程执行器
  │      └─ GPUModelRunner.execute_model()        vllm/v1/worker/gpu_model_runner.py
  │           ├─ 准备 InputBatch（谁在哪、block_table 长什么样）
  │           ├─ 准备 attention metadata（哪些位置能看哪些位置）
  │           └─ 跑模型 → 采样 → ModelRunnerOutput
  │
  └─ 3. 更新
       Scheduler.update_from_output()
         ├─ 把新采样的 token 写回 Request.output_token_ids
         ├─ 判断哪些请求已完成
         └─ 释放完成请求的 block
"""
print(CALL_CHAIN)

print("对照一下你在 lab 里写的：")
print("  EngineCore.step() → Scheduler.schedule() → ModelRunner.execute_model()")
print("  三步一模一样，只是真实版本在每一步里塞了更多工程细节。")
# %% [markdown]
# ## 三、三个必须读懂的"眼"
#
# 源码有几万行，但只有三个函数是真正的枢纽。**读懂它们，其余都是枝叶。**
#
# | 函数 | 为什么是关键 |
# |---|---|
# | `EngineCore.step()` | 主循环。看清它就知道整个系统的时间线 |
# | `Scheduler.schedule()` | 所有调度策略、优先级、chunked prefill 都在这里 |
# | `KVCacheManager.allocate_slots()` | 所有显存分配、前缀复用、抢占都在这里 |
#
# 第三个特别值得注意：**`allocate_slots` 返回失败，就是 vLLM 触发抢占（preemption）的时机。** 线上看到 `num_requests_waiting` 涨起来，根因往往就在这一行。
# %%
import inspect


def show_signature(module_path, class_name, method_name):
    """把某个方法的签名和 docstring 打出来。"""
    try:
        module = __import__(module_path, fromlist=[class_name])
        cls = getattr(module, class_name)
        fn = getattr(cls, method_name)
        print(f"\n{'=' * 72}")
        print(f"{class_name}.{method_name}{inspect.signature(fn)}")
        doc = inspect.getdoc(fn)
        if doc:
            print(f"\n  {doc.split(chr(10) + chr(10))[0][:400]}")
    except Exception as e:
        print(f"\n{'=' * 72}")
        print(f"{class_name}.{method_name}: 定位失败（{type(e).__name__}）")
        print(f'  用 rg 手动找：  rg "def {method_name}" {ROOT}')


show_signature("vllm.v1.core.sched.scheduler", "Scheduler", "schedule")
show_signature("vllm.v1.core.kv_cache_manager", "KVCacheManager", "allocate_slots")
show_signature("vllm.v1.core.kv_cache_manager", "KVCacheManager", "get_computed_blocks")
show_signature("vllm.v1.core.block_pool", "BlockPool", "get_new_blocks")
show_signature("vllm.v1.engine.core", "EngineCore", "step")

print()
print("读签名比读实现更重要——签名直接告诉你这个模块的输入输出和职责边界。")
print("比如 allocate_slots 的参数里有 num_new_tokens 和 num_new_computed_tokens，")
print("一眼就能看出「已算的」和「新算的」是分开处理的，这正是前缀缓存生效的地方。")
# %% [markdown]
# ## 四、阅读顺序（三轮，约 6 小时）
#
# ### 第一轮：找骨架（2 小时）
#
# 目标是把调用链画出来，不要求读懂实现细节。
#
# ```bash
# rg "def step" vllm/v1/engine/core.py
# rg "def schedule" vllm/v1/core/sched/scheduler.py
# rg "def execute_model" vllm/v1/worker/gpu_model_runner.py
# rg "class SchedulerOutput" -A 40 vllm/v1/core/sched/output.py
# ```
#
# **验收**：能白板画出 `step() → schedule() → execute_model() → update_from_output()`，并说出每步的输入输出。
#
# ### 第二轮：盯住三个"眼"（2.5 小时）
#
# ```bash
# rg "def schedule" -A 80 vllm/v1/core/sched/scheduler.py
# rg "def allocate_slots" -A 60 vllm/v1/core/kv_cache_manager.py
# rg "def get_computed_blocks" -A 40 vllm/v1/core/kv_cache_manager.py
# rg "def hash_block_tokens" -A 10 vllm/v1/core/kv_cache_utils.py
# rg "def cache_full_blocks" -A 40 vllm/v1/core/block_pool.py
# ```
#
# **验收**：能指出
#
# - chunked prefill 由哪一行产生（找 `max_num_batched_tokens` 和 `min(...)`）
# - 前缀缓存命中后 `num_computed_tokens` 在哪里被改写
# - 分配失败时抢占从哪里触发
#
# ### 第三轮：挑一个专题深挖（1.5 小时）
#
# 挑一个和你业务最相关的：
#
# ```bash
# rg "prefix" vllm/v1/core/ -l          # 前缀缓存
# rg "preempt" vllm/v1/core/sched/scheduler.py   # 抢占
# ls vllm/v1/spec_decode/               # 投机解码
# ls vllm/v1/attention/backends/        # attention 后端
# ```
#
# 这一轮的目标不是"读完"，而是**能在面试里就这个专题讲 5 分钟**。
# %% [markdown]
# ## 五、版本漂移：必须知道的事
#
# | 变化 | 影响 |
# |---|---|
# | V0 → V1 是重写 | 网上 V0 时代的源码分析**大部分已经失效**，看之前先确认版本 |
# | 路径会移动 | 判断依据：`rg "class Scheduler" vllm/` 永远比记路径可靠 |
# | `GPUModelRunner` 是 V1 的名字 | V0 里叫 `ModelRunner` |
# | 配置逐步集中到 `vllm/config.py` | 参数从裸参数变成 `SchedulerConfig` / `CacheConfig` 等 |
# | 部分能力被拆到独立项目 | PD 分离的 KV transfer、prefix-aware routing 等，有的已移出主仓库 |
#
# **一条实用建议**：面试时别说"vLLM 的 XX 文件第 N 行"，说"vLLM V1 里 `Scheduler.schedule()` 负责 XX，`KVCacheManager` 负责 YY"。**职责描述不会过时，行号和路径会。**
# %% [markdown]
# ## 六、面试怎么把源码讲出来
#
# 面试官不会考你背代码，他想确认的是：**你是真的理解这套系统的设计，还是只会用。** 三个层次：
#
# **第一层（会用）**：知道 vLLM 有 PagedAttention、continuous batching、prefix caching。
#
# **第二层（懂机制）**：能解释每个机制的收益来源和代价。（这是前 8 章的水平）
#
# **第三层（懂架构）**：能说出模块边界以及为什么这样划分。比如：
#
# > vLLM V1 把调度、显存、执行拆成 `Scheduler` / `KVCacheManager` / `ModelRunner` 三块，`EngineCore.step()` 是胶水。这个划分的价值在于变更隔离——换调度策略只动 `Scheduler`，换 attention 后端只动 `ModelRunner`，换显存回收策略只动 `KVCacheManager`。
# >
# > 举个具体例子：chunked prefill 听起来是个新特性，但实现上它只是 `Scheduler` 里 token 预算 `max_num_batched_tokens` 的一个取值——因为 `Request.num_computed_tokens` 已经把"算到哪儿了"这个状态表达清楚了，切块不需要任何额外机制。

print("第三层的答法为什么有说服力：")
print("  它证明你不只知道'有这个东西'，还知道'它为什么能这么简单地实现'。")
print("  后者门槛高得多——你得先理解数据结构的表达力，才能看出机制是自然的推论。")
# %% [markdown]
# ## 七、动手作业（做完才算真的读过）
#
# 光看不写，三天就忘。挑一个做：
# %%
HOMEWORK = """
① 加一行日志，把调度决策打出来
   在 Scheduler.schedule() 里插入 print，记录每轮 scheduled 的请求数和 token 数。
   跑一次真实推理，观察 chunked prefill 开/关时这一行的变化。
   最省力，也最能建立直觉。

② 让前缀缓存命中可见
   在 KVCacheManager.get_computed_blocks() 的返回处加日志，打印命中了多少个 block。
   然后跑第 09 章那个 prompt 结构 A/B 实验，看两种布局下命中数差多少。

③ 改调度策略
   给 Scheduler 加一个"短请求优先"策略（比如按 max_tokens 排序 waiting 队列），
   观察 TTFT p99 的变化。这是最常见的入门级贡献。

④ 提一个 PR
   从文档修正开始也行。走完一次完整的开源协作流程——issue、分支、CI、review——
   这件事本身就值钱，因为它证明你能在别人的代码库里工作。
"""
print(HOMEWORK)
# %% [markdown]
# ## 八、这一章之后
#
# 到这里，这个 lab 的使命就完成了。你现在应该能做到三件事：
#
# 1. **打开 vLLM 源码不迷路**：知道从 `EngineCore.step()` 进去，知道三个"眼"在哪。
# 2. **把机制和实现对上**：lab 里的 `KVCacheManager` 就是 vLLM 的 `KVCacheManager`，不是类比。
# 3. **在面试里讲架构而不只是讲概念**：能说出模块边界和划分理由。
#
# 最后提醒一句：**读完源码不等于理解源码。** 检验标准很简单——别人问你"如果让你改一个调度策略，你会动哪里、会不会影响别的地方"，你能不假思索地答出来，那就是真的读懂了。
#
# **配套文档**：完整的「lab 概念 ↔ vLLM 源码」对照表在 `docs/vllm-mapping.md`，面试前可以拿它快速过一遍。
