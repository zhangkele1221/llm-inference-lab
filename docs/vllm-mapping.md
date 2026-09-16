# lab 概念 ↔ vLLM 源码 对照表

这份表是**面试前的速查卡**。左边是你在 lab 里手写过的东西，右边是 vLLM V1 里对应的实现。

> **使用前提**：vLLM 版本迭代快，路径和类名会漂移。表里的路径以 V1 为主，实际使用时优先用 `rg` 定位。**职责描述不会过时，行号和路径会。**

---

## 一、三层架构

| lab | vLLM V1 | 找到它 |
|---|---|---|
| `EngineCore.step()` | `EngineCore.step()` | `rg "def step" vllm/v1/engine/core.py` |
| `Scheduler` | `Scheduler` | `vllm/v1/core/sched/scheduler.py` |
| `SchedulerOutput` | `SchedulerOutput` | `vllm/v1/core/sched/output.py` |
| `ModelRunner.execute_model()` | `GPUModelRunner.execute_model()` | `vllm/v1/worker/gpu_model_runner.py` |
| `KVCacheManager` | `KVCacheManager` | `vllm/v1/core/kv_cache_manager.py` |
| `BlockPool` | `BlockPool` | `vllm/v1/core/block_pool.py` |

**主循环（两边一模一样）：**

```
schedule()            决定这一轮跑什么
execute_model()       跑模型
update_from_output()  把结果写回请求状态
```

**职责边界（面试重点）：**

| 模块 | 管什么 | 不管什么 |
|---|---|---|
| `Scheduler` | 跑哪些请求、各跑几个 token | 不碰显存，不碰模型 |
| `KVCacheManager` | KV 放哪、能复用多少 | 不决定跑什么 |
| `ModelRunner` | 拼 batch、跑模型、采样 | 不决定调度策略 |

这样划分的价值是**变更隔离**：换调度策略只动 `Scheduler`，换 attention 后端只动 `ModelRunner`，换显存回收策略只动 `KVCacheManager`。

---

## 二、请求对象

| lab | vLLM | 说明 |
|---|---|---|
| `Request.request_id` | `Request.request_id` | 同名 |
| `Request.prompt_token_ids` | `Request.prompt_token_ids` | 同名 |
| `Request.output_token_ids` | `Request.output_token_ids` | 同名 |
| `Request.num_computed_tokens` | `Request.num_computed_tokens` | **语义完全一致，这是全表最重要的一个字段** |
| `Request.all_token_ids()` | `Request.all_token_ids` | vLLM 里是属性 |
| `Request.num_tokens_to_schedule()` | `Request.num_tokens_with_spec` 等 | vLLM 还要算投机解码的草稿 token |
| `Request.past`（本 lab 持有 KV 张量） | `Request.block_ids` + `block_table` | **这是最大的简化**：vLLM 只持 block 索引 |

### 为什么 `num_computed_tokens` 是钥匙

vLLM 用**一个字段**统一表达了三种场景：

| 场景 | 这个字段怎么变 |
|---|---|
| 正常 prefill | 一次性从 0 涨到 prompt 长度 |
| Chunked prefill | 一轮涨一点，分多轮涨到 prompt 长度 |
| 前缀缓存命中 | 入场时直接被推进到命中长度 |

**推论**：chunked prefill 不需要任何新机制，它只是这个字段加上一个 token 预算的自然结果。能讲清这一点，说明你理解的是设计而不是特性列表。

---

## 三、调度器

| lab | vLLM | 说明 |
|---|---|---|
| `Scheduler.waiting` | `Scheduler.waiting` | 同名，等待队列 |
| `Scheduler.running` | `Scheduler.running` | 同名，运行队列 |
| `Scheduler.finished` | `Scheduler.finished_req_ids` | vLLM 只存 id |
| — | `Scheduler.skipped_waiting` | vLLM 特有：本轮因显存不足没调度的请求 |
| `max_num_seqs` | `SchedulerConfig.max_num_seqs` | **同名参数，语义一致** |
| `max_num_batched_tokens` | `SchedulerConfig.max_num_batched_tokens` | **同名参数，chunked prefill 的开关** |
| `Scheduler.schedule()` | `Scheduler.schedule()` | vLLM 里还处理优先级、抢占、LoRA |
| `Scheduler.update_from_output()` | `Scheduler.update_from_output()` | 同名 |

**vLLM 比 lab 多出来的东西**（面试可能追问）：

- **优先级**：`priority` 参数 + `SchedulingPolicy`，FCFS 或优先级队列
- **抢占（preemption）**：显存不够时把部分 running 请求踢回 waiting，重新计算或换出 KV
- **`allocate_slots` 失败**：这是触发抢占的信号
- **投机解码**：调度时每个请求要多排草稿 token

---

## 四、KV cache 管理层

| lab | vLLM | 找到它 |
|---|---|---|
| `KVCacheBlock.block_id` | 同名字段 | `vllm/v1/core/kv_cache_utils.py` |
| `KVCacheBlock.ref_cnt` | 同名字段 | 同上 |
| `KVCacheBlock.block_hash` | 同名字段 | 同上 |
| `hash_block_tokens()` | `hash_block_tokens()` | 同上 |
| `BlockPool.get_new_blocks()` | `get_new_blocks()` | `vllm/v1/core/block_pool.py` |
| `BlockPool.free_blocks_of()` | `free_blocks()` | 同上 |
| `BlockPool.cached_block_hash_to_block` | 同名字段 | 同上 |
| `BlockPool.cache_full_block()` | `cache_full_blocks()` | 同上 |
| `BlockPool.free_blocks`（deque） | `FreeKVCacheBlockQueue` | 双向链表，O(1) 增删 |
| `KVCacheManager.allocate_slots()` | `allocate_slots()` | `vllm/v1/core/kv_cache_manager.py` |
| `KVCacheManager.attach_prefix_cache()` | `get_computed_blocks()` | 同上 |
| `KVCacheManager.cache_blocks()` | `cache_full_blocks()` | `vllm/v1/core/block_pool.py` |
| `KVCacheManager.free_request()` | `free()` | `vllm/v1/core/kv_cache_manager.py` |

**lab 里被简化掉、但真实 vLLM 有的**：

- **`KVCacheCoordinator`**：管理多组 KV（比如 sliding window + full attention 混合模型）
- **Prefix caching 的实际 KV 数据**：vLLM 的 block 指向真实显存，lab 只维护账本
- **写时复制（COW）**：共享 block 被覆写前要先摘除缓存哈希
- **淘汰队列**：引用计数归零但仍有哈希的 block 进入淘汰队列，前缀缓存和回收同时成立

---

## 五、执行层

| lab | vLLM | 说明 |
|---|---|---|
| `ModelRunner._run_decode_batch()` | `GPUModelRunner.execute_model()` | lab 用填充对齐，vLLM 用 block table |
| — | `InputBatch` | vLLM 用它维护 batch 的账本（谁在哪、block_table、采样参数） |
| — | `AttentionMetadataBuilder` | 为不同 attention 后端准备元数据 |
| — | CUDA graph / 编译 | 降低 kernel 启动开销 |
| — | `MultiprocExecutor` / `UniprocExecutor` | 多进程执行 |
| 位置偏移 + 逐序列掩码 | **block_table + attention metadata** | lab 的填充是 vLLM 明确要避免的 |

---

## 六、配置与参数

| lab | vLLM | 作用 |
|---|---|---|
| `max_num_seqs` | `SchedulerConfig.max_num_seqs` | 单批最多几条序列 |
| `max_num_batched_tokens` | `SchedulerConfig.max_num_batched_tokens` | 单批最多几个 token，控制 chunked prefill |
| `block_size` | `CacheConfig.block_size` | 默认 16 |
| `num_blocks` | 由 `gpu_memory_utilization` 和模型大小推算 | KV cache 能占多少显存 |
| — | `enable_prefix_caching` | 开关前缀缓存 |
| — | `--enable-chunked-prefill` | 开关 chunked prefill |
| — | `gpu_memory_utilization` | 允许 vLLM 用多少比例的显存 |

---

## 七、面试速答：把 lab 经验翻译成源码语言

| 面试官问 | 只说概念（第二层） | 带上源码（第三层） |
|---|---|---|
| continuous batching 收益来自哪 | 完成的请求立刻释放，新请求补位 | `Scheduler` 的 `running`/`waiting` 两个队列每轮重新组批；收益取决于输出长度方差 |
| PagedAttention 解决什么 | 减少显存碎片 | `BlockPool` 按 block 分配 + `block_table` 索引，消除外部碎片和对齐填充 |
| prefix caching 怎么实现 | 相同前缀复用 KV | 链式 `hash_block_tokens` + block `ref_cnt` 共享；命中体现在 `num_computed_tokens` 被直接推进 |
| chunked prefill 怎么做的 | 把长 prefill 切块 | `Scheduler` 的 `max_num_batched_tokens` 预算约束，**没有新机制** |
| 前缀缓存什么时候失效 | 前缀不一样就失效 | 链式哈希要求从第一个 block 起连续匹配，断开即全失效 |
| 显存不够会怎样 | 会排队 | `allocate_slots()` 失败 → 触发抢占（preemption） |
| 怎么排查吞吐不达标 | 看指标 | 先看 `num_requests_waiting` 是否为 0，区分"容量不足"和"单请求效率低" |

---

## 八、一页速记

```
EngineCore.step()
  ├─ Scheduler.schedule()               跑什么
  │    ├─ running 优先（decode 1 token / chunked prefill 剩余量）
  │    ├─ waiting 补位（prefill）
  │    ├─ 受 max_num_batched_tokens 约束  ← chunked prefill 的来源
  │    └─ KVCacheManager.allocate_slots()  KV 放哪
  │         └─ BlockPool.get_new_blocks()
  ├─ ModelRunner.execute_model()        怎么跑
  └─ Scheduler.update_from_output()     写回状态、释放 block
```

**三个"眼"**：`EngineCore.step()` / `Scheduler.schedule()` / `KVCacheManager.allocate_slots()`

**一个钥匙字段**：`Request.num_computed_tokens`

**一句能加分的总结**：vLLM V1 把调度、显存、执行拆成三块，价值在于变更隔离；而 chunked prefill、prefix caching 这些听起来独立的特性，本质上都是 `num_computed_tokens` 这个字段的不同取值方式。
