# LLM Inference Lab · 大模型推理基础设施动手实验室

面向**推理 infra / 推理平台**岗位面试的实战课程。十个 notebook，从"手写一个 GPT"开始，一路做到"多副本前缀感知路由网关"。每一章都跑在 Colab 免费 GPU 上，**零配置**。

---

## 一、在 Colab 里打开（重点）

把 GitHub 链接里的 `github.com` 换成 `colab.research.google.com/github` 即可直接加载：

```
原始链接： https://github.com/<你的用户名>/llm-inference-lab/blob/main/notebooks/01_env_and_benchmark.ipynb
Colab 版：  https://colab.research.google.com/github/<你的用户名>/llm-inference-lab/blob/main/notebooks/01_env_and_benchmark.ipynb
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
| 04 | [Continuous Batching](notebooks/04_continuous_batching.ipynb) | 从零实现迭代级调度器，对比 static batching 的吞吐差距 | 收益来源、代价、什么时候收益最大 | 2h |
| 05 | [Paged KV Cache 与前缀复用](notebooks/05_paged_kv_prefix_cache.ipynb) | 实现 block manager + 引用计数前缀共享，验证输出一致性 | block size 权衡、prefix cache 命中与失效 | 2.5h |
| 06 | [Chunked Prefill 与尾延迟](notebooks/06_chunked_prefill.ipynb) | 把长 prefill 切块并与 decode 混批，测 TBT p99 的改善 | 为什么改善尾延迟，单请求 TTFT 的代价 | 1.5h |
| 07 | [投机解码](notebooks/07_speculative_decoding.ipynb) | 实现 draft + 验证的投机解码，扫出负收益区域 | 什么条件下投机解码是负收益 | 2h |
| 08 | [量化](notebooks/08_quantization.ipynb) | 手写 INT8 权重量化与 KV cache 量化，量化精度损失 | FP8/INT4 取舍、KV cache 为什么更敏感 | 1.5h |
| 09 | [vLLM 实战](notebooks/09_vllm_benchmark.ipynb) | 真实 vLLM 部署 + 压测 + 指标解读，验证 prompt 结构对缓存命中率的影响 | 现场定位吞吐不达标的方法论 | 2h |
| 10 | [毕业项目：前缀感知路由](notebooks/10_capstone_prefix_routing.ipynb) | 一个多副本网关，对比轮询/最小负载/前缀亲和三种路由策略 | 你简历上那个项目的原型 | 3h |

**总计约 18.5 小时**，按每天 1.5-2 小时算，两周走完。

---

## 三、为什么自己写一个模型，而不是直接调 vLLM

| | 直接调 vLLM | 本仓库的做法 |
|---|---|---|
| 能看 KV cache 怎么分配吗 | 不能，黑盒 | 能，自己写 block manager |
| 能改调度策略吗 | 不能 | 能，自己写 scheduler |
| 免费 Colab 跑得动吗 | T4 勉强，且启动慢 | 27M 参数，秒级启动 |
| 能验证"前缀复用不改变输出"吗 | 只能看指标 | 能逐位对齐 logits 验证 |

手写一遍之后再去看 vLLM 源码，你看的是"它怎么实现我已知的东西"，而不是"一堆陌生概念"。第 09 章会切回真实的 vLLM 做压测——两条路都走过，面试时你讲的东西才有质感。

**关于 MiniGPT**：它不是玩具。结构和 Llama 同源（pre-norm、因果注意力、MLP 4 倍扩张、权重共享），只是去掉了 RoPE 换成可学习位置编码，并且支持**任意位置偏移的 KV cache**——这正是实现连续批处理的关键。

---

## 四、本地运行

```bash
git clone https://github.com/<你的用户名>/llm-inference-lab.git
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
│   └── study-plan.md          # 两周学习计划 + 每周自检
├── notebooks/                 # 十个 Notebook（Colab 直接打开这些）
├── tools/
│   ├── build_notebooks.py     # 维护用：把 nbsrc/*.py 编译成 ipynb
│   └── nbsrc/                 # Notebook 的源码（percent 格式）
└── requirements.txt
```

> `tools/` 只是给维护用的。你日常只需要动 `notebooks/`。如果想改课程内容，改 `tools/nbsrc/*.py` 然后跑 `python tools/build_notebooks.py`，会重新生成全部 ipynb。

---

## 六、学完你应该能回答的问题

- continuous batching 相比 static batching，收益具体来自哪里？代价是什么？
- PagedAttention 的 block size 怎么选？大了小了分别出什么问题？
- prefix cache 在什么场景下会完全失效？
- chunked prefill 为什么能改善尾延迟？它对单请求 TTFT 做了什么牺牲？
- 投机解码在什么条件下是**负收益**？
- 给你模型规格和卡型，当场算出最大并发数。
- 一个 vLLM 部署吞吐只有预期的一半，你怎么定位？

如果这七个问题你都能讲满三分钟并且扛得住追问，这个仓库的使命就完成了。

---

## License

MIT
