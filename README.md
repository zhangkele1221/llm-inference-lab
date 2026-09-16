# LLM Inference Lab · 大模型推理基础设施动手实验室

面向**推理 infra / 推理平台**岗位面试的实战课程。十一个 notebook，从"手写一个 GPT"开始，一路做到"多副本前缀感知路由网关"。每一章都跑在 Colab 免费 GPU 上，**零配置**。

**核心设计：所有组件的命名、职责划分和主循环结构都刻意对齐 vLLM V1。** 学完之后你打开 `vllm/v1/core/block_pool.py`，看到的应该是一堆熟悉的东西——而不是一堆陌生概念。

```
EngineCore.step()          ← 本仓库同名类，对应 vllm/v1/engine/core.py
  ├─ Scheduler.schedule()  ← 对应 vllm/v1/core/sched/scheduler.py
  ├─ ModelRunner.execute_model()
  └─ Scheduler.update_from_output()
```

完整的「lab 概念 ↔ vLLM 源码」对照表见 [docs/vllm-mapping.md](docs/vllm-mapping.md)。

---

## 一、在 Colab 里打开（重点）

把 GitHub 链接里的 `github.com` 换成 `colab.research.google.com/github` 即可直接加载：

```
原始链接： https://github.com/zhangkele1221/llm-inference-lab/blob/main/notebooks/01_env_and_benchmark.ipynb
Colab 版：  https://colab.research.google.com/github/zhangkele1221/llm-inference-lab/blob/main/notebooks/01_env_and_benchmark.ipynb
```

**两个前提：**

1. 仓库必须是 **public**，否则 Colab 会要求额外授权。
2. 打开后先确认 **运行时 → 更改运行时类型 → 硬件加速器选 GPU**，否则跑的是 CPU（第 09、10 章会直接失败）。

> 小技巧：打开一次后，点 Colab 左上角的 **"在云端硬盘中保存副本"**，之后可以直接从 Colab 的"最近"里打开，不用每次拼 URL。

---

## 二、课程地图

| # | Notebook | 你会亲手做出什么 | 对应面试考点 | 预计 |
|---|---|---|---|---|
| 01 | [环境与基准工具](notebooks/01_env_and_benchmark.ipynb) | 一个可复用的推理基准测试框架，和一个能塞进 16G 显存的 MiniGPT | 延迟/吞吐的正确测法、TTFT 与 TPOT 的区别 | 1h |
| 02 | [Prefill vs Decode](notebooks/02_prefill_vs_decode.ipynb) | 实测两种阶段的算力/带宽利用率，验证 arithmetic intensity 理论 | 为什么 prefill 看 MFU、decode 看 MBU | 1.5h |
| 03 | [KV Cache 与显存容量](notebooks/03_kv_cache_capacity.ipynb) | 一个显存容量计算器，给定模型和卡型当场算出最大并发 | **必考题**：KV cache 公式、GQA 的影响 | 1.5h |
| 04 | [vLLM 的调度循环](notebooks/04_continuous_batching.ipynb) | 跑通 `EngineCore.step()` 三段式主循环，只改一个 `schedule()` 就做出 static batching 对照组 | 收益来源、代价、调度与执行为什么要分离 | 2h |
| 05 | [PagedAttention 与前缀缓存](notebooks/05_paged_kv_prefix_cache.ipynb) | 实现 `KVCacheBlock` / `BlockPool` / `KVCacheManager`，看前缀命中如何直接推进 `num_computed_tokens` | block size 权衡、引用计数共享、命中与失效规律 | 2.5h |
| 06 | [Chunked Prefill 与尾延迟](notebooks/06_chunked_prefill.ipynb) | 把 token 预算从 4096 调到 256，看长 prompt 如何**自动**被切块 | 尾延迟的重新分配、它为什么不是独立机制 | 1.5h |
| 07 | [投机解码](notebooks/07_speculative_decoding.ipynb) | 实现 draft + 验证的投机解码，扫出负收益区域 | 什么条件下投机解码是负收益 | 2h |
| 08 | [量化](notebooks/08_quantization.ipynb) | 手写 INT8 权重量化与 KV cache 量化，量化精度损失 | FP8/INT4 取舍、KV cache 为什么更敏感 | 1.5h |
| 09 | [vLLM 实战](notebooks/09_vllm_benchmark.ipynb) | 真实 vLLM 部署 + 压测 + 指标解读，验证 prompt 结构对缓存命中率的影响 | 现场定位吞吐不达标的方法论 | 2h |
| 10 | [毕业项目：前缀感知路由](notebooks/10_capstone_prefix_routing.ipynb) | 一个多副本网关，对比轮询/最小负载/前缀亲和三种路由策略 | 你简历上那个项目的原型 | 3h |
| 11 | [vLLM 源码导读](notebooks/11_vllm_source_tour.ipynb) | 在你自己的环境里定位源码、打印关键函数签名、三轮阅读路线 | 面试时怎么把机制讲成架构 | 1.5h |

**总计约 20 小时**，按每天 1.5-2 小时算，两周走完。

---

## 三、为什么自己写一个模型，而不是直接调 vLLM

| | 直接调 vLLM | 本仓库的做法 |
|---|---|---|
| 能看 KV cache 怎么分配吗 | 不能，黑盒 | 能，自己写 `BlockPool` |
| 能改调度策略吗 | 不能 | 能，自己写 `Scheduler.schedule()` |
| 免费 Colab 跑得动吗 | T4 勉强，且启动慢 | 27M 参数，秒级启动 |
| 能验证"前缀复用不改变输出"吗 | 只能看指标 | 能逐位对齐 logits 验证 |

关键不在于"自己写"，而在于**用和 vLLM 一样的架构去写**。所以：

- 请求对象叫 `Request`，带 `num_computed_tokens` 字段——和 vLLM 语义完全一致
- 调度器叫 `Scheduler`，有 `waiting` / `running` 两个队列——和 vLLM 同名
- KV 管理层拆成 `KVCacheBlock` / `BlockPool` / `KVCacheManager`——和 vLLM 同样的三级结构
- 主循环是 `EngineCore.step()` 的 schedule → execute → update——和 vLLM 同构

**这样你只需要建立一套心智模型。** 第 09 章切回真实 vLLM 做压测，第 11 章直接带你读源码——三条路径指向同一套概念。

**关于 MiniGPT**：它不是玩具。结构和 Llama 同源（pre-norm、因果注意力、MLP 4 倍扩张、权重共享），只是去掉了 RoPE 换成可学习位置编码，并且支持**任意位置偏移的 KV cache**——这正是实现连续批处理的关键。

---

## 四、本地运行

```bash
git clone https://github.com/zhangkele1221/llm-inference-lab.git
cd llm-inference-lab
pip install -r requirements.txt
jupyter lab notebooks/
```

没有 GPU 也能跑第 01 至 08 章（会慢一些，代码里已经做了 CPU 兜底）；第 09、10 章需要 GPU。

---

## 五、仓库结构

```
llm-inference-lab/
├── README.md                  # 你在这里
├── docs/
│   ├── study-plan.md          # 两周学习计划 + 每日验收标准
│   └── vllm-mapping.md        # lab 概念 ↔ vLLM 源码对照表（面试速查卡）
├── notebooks/                 # 十一个 Notebook（Colab 直接打开这些）
├── tools/
│   ├── build_notebooks.py     # 维护用：把 nbsrc/*.py 编译成 ipynb
│   └── nbsrc/                 # Notebook 的源码（percent 格式）
└── requirements.txt
```

> `tools/` 只是给维护用的。你日常只需要动 `notebooks/`。如果想改课程内容，改 `tools/nbsrc/*.py` 然后跑 `python tools/build_notebooks.py`，会重新生成全部 ipynb。

> 十个 notebook 共用的「引导单元」（MiniGPT + 测量工具 + vLLM 形状的推理核心）只在 `tools/build_notebooks.py` 里维护一份，编译时注入到每个 notebook，所以每章都能独立运行又不会出现副本漂移。

---

## 六、学完你应该能回答的问题

**机制层**

- continuous batching 相比 static batching，收益具体来自哪里？代价是什么？
- PagedAttention 的 block size 怎么选？大了小了分别出什么问题？
- prefix cache 在什么场景下会完全失效？
- chunked prefill 为什么能改善尾延迟？它对单请求 TTFT 做了什么牺牲？
- 投机解码在什么条件下是**负收益**？
- 给你模型规格和卡型，当场算出最大并发数。
- 一个 vLLM 部署吞吐只有预期的一半，你怎么定位？

**架构层（这一层才拉开差距）**

- vLLM 为什么把调度、显存、执行拆成三个模块？各自的边界在哪？
- `Request.num_computed_tokens` 这一个字段，怎么同时表达 prefill、chunked prefill 和前缀缓存命中？
- chunked prefill 为什么不需要新机制？它落在哪一行代码上？
- 如果让你改一个调度策略，你会动哪里、会不会影响别的地方？
- 显存不足时 vLLM 会发生什么？触发点在哪个函数的返回值上？

如果这两组问题你都能讲满三分钟并且扛得住追问，这个仓库的使命就完成了。

---

## License

MIT
