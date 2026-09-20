#!/usr/bin/env python3
"""把 tools/nbsrc/*.py（jupytext percent 格式）编译成 notebooks/*.ipynb。

用法：
    python tools/build_notebooks.py

为什么要有这个脚本：
    1. 十个 notebook 都需要同一份「引导单元」（MiniGPT + 测量工具）。如果手工复制，
       改一处就要改十处。这里用 @@SETUP@@ 占位符统一注入，保证十个 notebook
       各自自包含（Colab 零配置），同时只有一个真实来源。
    2. percent 格式比 ipynb 的 JSON 更适合写和 review，diff 也干净。

注意：notebooks/*.ipynb 是生成物，但会提交到仓库。日常学习直接改 notebook 也行，
      只是下次重新编译会被覆盖。
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "tools" / "nbsrc"
OUT_DIR = ROOT / "notebooks"

CELL_MARK = re.compile(r"^#\s*%%(\s*\[markdown\])?\s*$")
SETUP_MARK = "@@SETUP@@"


# --------------------------------------------------------------------------
# 引导单元：所有 notebook 共用，由 @@SETUP@@ 占位符注入
# --------------------------------------------------------------------------
SETUP_CODE = r'''
# ===== 引导单元：环境检查 + 测量工具 + MiniGPT（每章自带，直接运行）=====
# 说明：本单元在每个 notebook 里都有一份完整副本，目的是让任何一个 notebook
#       都能在 Colab 里零配置独立运行。想改模型结构，请改 tools/build_notebooks.py
#       里的 SETUP_CODE，然后重跑编译脚本。
#
# 架构对齐：下面这套推理核心刻意模仿了 vLLM V1 的模块划分与命名，
#   详见 docs/vllm-mapping.md 的对照表。
#       EngineCore.step()           ←→ vllm/v1/engine/core.py
#         ├─ Scheduler.schedule()   ←→ vllm/v1/core/sched/scheduler.py
#         ├─ ModelRunner.execute_model() ←→ vllm/v1/worker/gpu_model_runner.py
#         └─ Scheduler.update_from_output()
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# MiniGPT 只有 2700 万参数，用 float16 跑在 GPU 上；CPU 上 float16 很慢，用 float32
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# 常见卡的关键参数（近似值）。如果你的卡不在表里，直接在这里补一行：
#   "你的卡型号": {"mem_gb": .., "bw_gbps": .., "fp16_tflops": .., "arch": ".."},
# 三个数字都能在厂商 datasheet 上查到。第 02、03 章会用到它们。
CARD_SPECS = {
    "Tesla T4":        {"mem_gb": 16, "bw_gbps": 320,  "fp16_tflops": 65,  "arch": "Turing sm75"},
    "Tesla V100":      {"mem_gb": 16, "bw_gbps": 900,  "fp16_tflops": 125, "arch": "Volta sm70"},
    "A100-SXM4-40GB":  {"mem_gb": 40, "bw_gbps": 1555, "fp16_tflops": 312, "arch": "Ampere sm80"},
    "A100-SXM4-80GB":  {"mem_gb": 80, "bw_gbps": 2039, "fp16_tflops": 312, "arch": "Ampere sm80"},
    "L4":              {"mem_gb": 24, "bw_gbps": 300,  "fp16_tflops": 121, "arch": "Ada sm89"},
    "A10G":            {"mem_gb": 24, "bw_gbps": 600,  "fp16_tflops": 125, "arch": "Ampere sm86"},
    "H100 PCIe":       {"mem_gb": 80, "bw_gbps": 2000, "fp16_tflops": 756, "arch": "Hopper sm90"},
    "H100 80GB HBM3":  {"mem_gb": 80, "bw_gbps": 3350, "fp16_tflops": 989, "arch": "Hopper sm90"},
}


def lookup_card():
    """按 GPU 名称匹配规格表。匹配不到就返回零值，提醒你手工补。"""
    if not torch.cuda.is_available():
        return {"name": "CPU", "mem_gb": 0, "bw_gbps": 0, "fp16_tflops": 0, "arch": "CPU"}
    name = torch.cuda.get_device_properties(0).name
    for key, spec in CARD_SPECS.items():
        # 双向包含匹配：Colab 可能报 "Tesla T4"，也可能报 "NVIDIA L4"
        if key.lower() in name.lower() or name.lower().replace("nvidia ", "") in key.lower():
            return {"name": name, **spec}
    return {
        "name": name,
        "mem_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 1),
        "bw_gbps": 0,
        "fp16_tflops": 0,
        "arch": "未知卡型 → 请查 datasheet 后补进 CARD_SPECS",
    }


SPEC = lookup_card()


def sync():
    """GPU 是异步执行的，计时前必须同步，否则测到的是下发时间不是执行时间。"""
    if DEVICE == "cuda":
        torch.cuda.synchronize()


def bench(fn, warmup=3, iters=10):
    """返回单次调用的平均耗时（毫秒）。warmup 用来排除首次 kernel 编译等开销。"""
    for _ in range(warmup):
        fn()
    sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    sync()
    return (time.perf_counter() - t0) / iters * 1000.0


def peak_mem_mb():
    """当前 CUDA 峰值显存占用（MB）。"""
    if DEVICE != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated() / 1024 ** 2


def reset_peak():
    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()


class Config:
    def __init__(self, vocab_size=50257, block_size=1024, n_layer=4, n_head=6, n_embd=384):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head


class CausalSelfAttention(nn.Module):
    """因果自注意力，支持 KV cache。

    past_kv 传入历史的 (k, v)，本步只为新 token 计算 Q/K/V，然后拼在历史后面。
    返回 (输出, 更新后的 (k, v))，其中 k/v 的 shape 是 (B, n_head, 总长度, head_dim)。
    """

    def __init__(self, cfg):
        super().__init__()
        self.n_head = cfg.n_head
        self.head_dim = cfg.head_dim
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)

    def forward(self, x, past_kv=None, attn_mask=None):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)

        S = k.size(2)  # 总长度 = 历史 + 本步新增
        if attn_mask is None:
            # 默认因果掩码：本步第 i 个 query 的绝对位置是 S-T+i，只能看见 <= 它的 key
            mask = torch.ones(T, S, device=x.device).tril(diagonal=S - T).bool()
        else:
            # 外部传入的掩码，用于一个 batch 里混合不同进度的序列（第 04、06 章）
            mask = attn_mask
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y), (k, v)


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=False)

    def forward(self, x):
        return self.proj(F.gelu(self.fc(x)))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x, past_kv=None, attn_mask=None):
        h, present = self.attn(self.ln_1(x), past_kv, attn_mask)
        x = x + h
        x = x + self.mlp(self.ln_2(x))
        return x, present


class MiniGPT(nn.Module):
    """极简 GPT，结构与 Llama 同源：pre-norm + 因果注意力 + 4 倍扩张 MLP + 权重共享。

    与 Llama 的两处差异：
      - 用可学习位置编码代替 RoPE（简化实现，不影响调度实验的结论）
      - 没有 GQA（本仓库是 MHA，第 03 章会手工比较两者的 KV cache 大小）
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight  # 权重共享，省一份 embedding 参数

        def init(m):
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

        self.apply(init)

    def forward(self, idx, past_kvs=None, pos_offset=0, attn_mask=None):
        """idx: (B, T) 的 token id。

        past_kvs: 长度等于层数的列表，每项是 (k, v)；None 表示从零开始（prefill）。
        pos_offset: 本次输入的第一个 token 的绝对位置。传 int 表示整个 batch 用同一个
                    偏移；传 shape (B,) 的张量表示每条序列各用各的偏移——当 batch 里
                    混合了不同进度的请求时必须这样传。
        attn_mask: 可选的自定义注意力掩码，用于屏蔽填充位。
        """
        B, T = idx.shape
        if torch.is_tensor(pos_offset):
            pos = pos_offset.view(B, 1) + torch.arange(T, device=idx.device)[None, :]
        else:
            pos = torch.arange(pos_offset, pos_offset + T, device=idx.device)[None, :].expand(B, T)
        x = self.wte(idx) + self.wpe(pos)

        presents = []
        for i, blk in enumerate(self.blocks):
            past = None if past_kvs is None else past_kvs[i]
            x, present = blk(x, past, attn_mask)
            presents.append(present)
        return self.lm_head(self.ln_f(x)), presents

    @property
    def n_params(self):
        return sum(p.numel() for p in self.parameters())


def build_model(seed=0, device=DEVICE, dtype=DTYPE, **kw):
    torch.manual_seed(seed)
    cfg = Config(**kw)
    model = MiniGPT(cfg).to(device=device, dtype=dtype)
    return model.eval()


@torch.no_grad()
def generate_naive(model, idx, max_new_tokens):
    """不用 KV cache：每一步都把完整序列重新算一遍（O(n^2) 重算）。"""
    for _ in range(max_new_tokens):
        logits, _ = model(idx[:, -model.cfg.block_size:])
        idx = torch.cat([idx, logits[:, -1].argmax(-1, keepdim=True)], dim=1)
    return idx


@torch.no_grad()
def generate_cached(model, idx, max_new_tokens):
    """用 KV cache：prompt 只 prefill 一次，之后每步只喂 1 个 token。"""
    logits, past = model(idx)
    nxt = logits[:, -1].argmax(-1, keepdim=True)
    out = [nxt]
    pos = idx.size(1)
    for _ in range(max_new_tokens - 1):
        logits, past = model(nxt, past_kvs=past, pos_offset=pos)
        pos += 1
        nxt = logits[:, -1].argmax(-1, keepdim=True)
        out.append(nxt)
    return torch.cat([idx] + out, dim=1)


def kv_bytes(n_layer, n_kv_head, head_dim, seq_len, batch=1, dtype_bytes=2):
    """KV cache 字节数。注意是 2（K 和 V 各一份）。"""
    return 2 * n_layer * n_kv_head * head_dim * seq_len * batch * dtype_bytes


# ========== 以下是模仿 vLLM V1 架构的推理核心 ==========


class Request:
    """对应 vllm/v1/request.py 的 Request。

    num_computed_tokens 是 vLLM 里最核心的一个字段：它记录这条请求已经有
    多少 token 的 KV 被算过。prefill、chunked prefill、前缀缓存命中——
    三种看起来完全不同的场景，在 vLLM 里都只是「把 num_computed_tokens 往前推」。
    理解这一点，chunked prefill 就不再是独立机制，而是这个字段的自然结果。
    """

    def __init__(self, request_id, prompt_token_ids, max_tokens):
        self.request_id = request_id
        self.prompt_token_ids = list(prompt_token_ids)
        self.max_tokens = max_tokens
        self.output_token_ids = []
        self.num_computed_tokens = 0
        self.status = "waiting"      # waiting / running / finished
        # 本仓库简化：直接把 KV 张量挂在请求上。
        # 真实 vLLM 不这么做——请求只持有 block_table，物理 block 由 KVCacheManager 管（第 05 章）。
        self.past = None

    @property
    def num_prompt_tokens(self):
        return len(self.prompt_token_ids)

    def all_token_ids(self):
        return self.prompt_token_ids + self.output_token_ids

    def num_tokens_to_schedule(self):
        """还欠多少 token 没算：prefill 阶段是剩余 prompt 长度，decode 阶段是 1。"""
        if self.num_computed_tokens < self.num_prompt_tokens:
            return self.num_prompt_tokens - self.num_computed_tokens
        return 1

    @property
    def is_finished(self):
        return len(self.output_token_ids) >= self.max_tokens

    def __repr__(self):
        return (f"Request({self.request_id}, computed={self.num_computed_tokens}"
                f"/{self.num_prompt_tokens}, out={len(self.output_token_ids)}"
                f"/{self.max_tokens}, {self.status})")


class SchedulerOutput:
    """对应 vllm/v1/core/sched/output.py 的 SchedulerOutput。

    调度与执行之间唯一的接口。真实 vLLM 里这个结构还包含 block 分配结果、
    抢占列表等字段，这里只保留最必要的两个。
    """

    def __init__(self, scheduled_reqs, num_scheduled_tokens):
        self.scheduled_reqs = scheduled_reqs
        self.num_scheduled_tokens = num_scheduled_tokens   # {request_id: n}

    def __len__(self):
        return len(self.scheduled_reqs)


class Scheduler:
    """对应 vllm/v1/core/sched/scheduler.py 的 Scheduler。

    职责边界是这个架构里最值得学的一点：Scheduler 只决定
    「这一轮跑哪些请求、各自跑几个 token」，它既不碰显存也不碰模型。

        显存分配 → KVCacheManager（第 05 章）
        真正计算 → ModelRunner

    三个模块分离，才能各自独立替换实现。面试被问「说说 vLLM 的架构」时，
    先把这个职责划分讲清楚，比背模块名有用得多。
    """

    def __init__(self, max_num_seqs=8, max_num_batched_tokens=2048):
        self.waiting = []
        self.running = []
        self.finished = []
        self.max_num_seqs = max_num_seqs
        # 这个预算就是 chunked prefill 的开关：调小它，长 prompt 自然被切成多轮（第 06 章）
        self.max_num_batched_tokens = max_num_batched_tokens
        self.step_id = 0

    def add_request(self, req):
        self.waiting.append(req)

    def has_unfinished(self):
        return bool(self.waiting or self.running)

    def schedule(self):
        scheduled, num_tokens = [], {}
        budget = self.max_num_batched_tokens

        # 第一优先：正在跑的请求。已进 decode 的排 1 个 token；
        # 还在做 chunked prefill 的按剩余量排，但受 budget 限制。
        for req in list(self.running):
            if budget <= 0 or len(scheduled) >= self.max_num_seqs:
                break
            n = min(req.num_tokens_to_schedule(), budget)
            scheduled.append(req)
            num_tokens[req.request_id] = n
            budget -= n

        # 第二优先：从队列里补新请求进来做 prefill
        for req in list(self.waiting):
            if budget <= 0 or len(scheduled) >= self.max_num_seqs:
                break
            n = min(req.num_tokens_to_schedule(), budget)
            scheduled.append(req)
            num_tokens[req.request_id] = n
            budget -= n
            self.waiting.remove(req)
            req.status = "running"
            self.running.append(req)

        self.step_id += 1
        return SchedulerOutput(scheduled, num_tokens)

    def update_from_output(self, sched_out, sampled):
        """对应 vLLM 的 update_from_output：写回采样结果，处理完成与回收。

        本轮被调度但没产生 token 的请求（比如 chunked prefill 的中间块）
        不会出现在 sampled 里，它们保持 running，下一轮继续。
        """
        for req in sched_out.scheduled_reqs:
            if req.request_id not in sampled:
                continue
            req.output_token_ids.append(sampled[req.request_id])
            if req.is_finished:
                req.status = "finished"
                if req in self.running:
                    self.running.remove(req)
                self.finished.append(req)
                req.past = None      # 简化回收；真实 vLLM 走 KVCacheManager.free()


class ModelRunner:
    """对应 vllm/v1/worker/gpu_model_runner.py 的 GPUModelRunner。

    职责：把 Scheduler 排好的一批请求拼成一次前向，返回新采样的 token。

    与真实 vLLM 的差距（要如实知道）：
      · vLLM 用 block_table 让每条序列的 KV 物理上不连续，所以不需要填充；
        这里用「右填充 + 逐序列掩码」对齐，会浪费显存——第 05 章解决。
      · vLLM 会把 prefill 和 decode 混在同一个 batch 里跑；这里分成两组处理，
        纯粹是为了让代码可读，结论不受影响。
      · 输入准备、CUDA graph、attention metadata 这些都被省掉了。
    """

    def __init__(self, model):
        self.model = model

    @torch.no_grad()
    def _run_decode_batch(self, reqs):
        """把一批进度不同的 decode 请求拼成一次前向。"""
        B = len(reqs)
        lens = [r.num_computed_tokens for r in reqs]
        Lmax = max(lens)
        n_layer = self.model.cfg.n_layer

        padded = []
        for layer in range(n_layer):
            ks, vs = [], []
            for r in reqs:
                k, v = r.past[layer]
                pad = Lmax - k.size(2)
                if pad:
                    k = F.pad(k, (0, 0, 0, pad))
                    v = F.pad(v, (0, 0, 0, pad))
                ks.append(k)
                vs.append(v)
            padded.append((torch.cat(ks, 0), torch.cat(vs, 0)))

        # 逐序列掩码：真实历史 [0, L_i) + 新 token 落在下标 Lmax
        S = Lmax + 1
        mask = torch.zeros(B, 1, 1, S, dtype=torch.bool, device=DEVICE)
        for i, r in enumerate(reqs):
            mask[i, 0, 0, : lens[i]] = True
            mask[i, 0, 0, Lmax] = True

        ids = torch.tensor([[r.all_token_ids()[r.num_computed_tokens]] for r in reqs],
                           device=DEVICE)
        pos = torch.tensor(lens, device=DEVICE)
        logits, past = self.model(ids, past_kvs=padded, pos_offset=pos, attn_mask=mask)

        sampled = {}
        for i, r in enumerate(reqs):
            rebuilt = []
            for layer in range(n_layer):
                k_all, v_all = past[layer]
                k = torch.cat([k_all[i:i + 1, :, : lens[i]],
                               k_all[i:i + 1, :, Lmax:Lmax + 1]], dim=2)
                v = torch.cat([v_all[i:i + 1, :, : lens[i]],
                               v_all[i:i + 1, :, Lmax:Lmax + 1]], dim=2)
                rebuilt.append((k, v))
            r.past = rebuilt
            r.num_computed_tokens += 1
            sampled[r.request_id] = int(logits[i, -1].argmax(-1).item())
        return sampled

    @torch.no_grad()
    def execute_model(self, sched_out):
        decode_reqs, prefill_reqs = [], []
        for r in sched_out.scheduled_reqs:
            # 判断依据是「prompt 算完了没有」，而不是「本轮排了几个 token」
            if r.num_computed_tokens >= r.num_prompt_tokens:
                decode_reqs.append(r)
            else:
                prefill_reqs.append(r)

        sampled = {}
        if decode_reqs:
            sampled.update(self._run_decode_batch(decode_reqs))

        for r in prefill_reqs:
            n = sched_out.num_scheduled_tokens[r.request_id]
            start = r.num_computed_tokens
            chunk = r.all_token_ids()[start:start + n]
            toks = torch.tensor([chunk], device=DEVICE)
            logits, past = self.model(toks, past_kvs=r.past, pos_offset=start)
            r.past = past
            r.num_computed_tokens += len(chunk)
            # 只有 prompt 全部算完，才能采样第一个输出 token
            if r.num_computed_tokens >= r.num_prompt_tokens:
                sampled[r.request_id] = int(logits[:, -1].argmax(-1).item())
        return sampled


class EngineCore:
    """对应 vllm/v1/engine/core.py 的 EngineCore。

    整个 vLLM 的推理服务就跑在这三步上：

        schedule()            决定这一轮跑什么
        execute_model()       跑模型
        update_from_output()  把结果写回请求状态

    读懂这个循环你就抓住了 vLLM 的主干。后面所有优化——chunked prefill、
    前缀缓存、抢占、投机解码——都是在这三步里插桩。
    """

    def __init__(self, model, scheduler=None):
        self.scheduler = scheduler or Scheduler()
        self.runner = ModelRunner(model)
        self.step_id = 0
        self.steps = 0

    def step(self):
        sched_out = self.scheduler.schedule()
        if len(sched_out) == 0:
            return None
        sampled = self.runner.execute_model(sched_out)
        self.scheduler.update_from_output(sched_out, sampled)
        self.step_id += 1
        self.steps += 1
        return sampled

    def run(self, max_steps=10000):
        while self.scheduler.has_unfinished() and self.steps < max_steps:
            self.step()
        return self.steps


print(f"引导单元加载完成 | device={DEVICE} dtype={DTYPE} torch={torch.__version__}")
# ===== 引导单元结束 =====
'''.strip("\n")


def strip_md_hashes(text: str) -> str:
    """markdown cell 在源码里每行以 '# ' 开头，这里去掉，还原成正常 markdown。"""
    out = []
    for line in text.splitlines():
        if line.startswith("# "):
            out.append(line[2:])
        elif line.strip() == "#":
            out.append("")
        else:
            out.append(line)
    return "\n".join(out)


def parse_percent(text: str):
    """解析 percent 格式，返回 [(cell_type, body), ...]"""
    cells, ctype, buf = [], None, []
    for line in text.splitlines():
        m = CELL_MARK.match(line)
        if m:
            if ctype is not None:
                cells.append((ctype, "\n".join(buf).strip("\n")))
            ctype = "markdown" if m.group(1) else "code"
            buf = []
        elif ctype is not None:
            buf.append(line)
    if ctype is not None:
        cells.append((ctype, "\n".join(buf).strip("\n")))

    result = []
    for ctype, body in cells:
        if not body.strip():
            continue
        if SETUP_MARK in body:
            body = SETUP_CODE
        elif ctype == "markdown":
            body = strip_md_hashes(body)
        result.append((ctype, body.strip("\n")))
    return result


def to_ipynb(cells) -> dict:
    nb_cells = []
    for ctype, body in cells:
        lines = body.split("\n")
        source = [ln + "\n" for ln in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
        if ctype == "markdown":
            nb_cells.append({"cell_type": "markdown", "metadata": {}, "source": source})
        else:
            nb_cells.append(
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": source,
                }
            )
    return {
        "cells": nb_cells,
        "metadata": {
            "colab": {"provenance": [], "toc_visible": True},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }


def main() -> int:
    if not SRC_DIR.is_dir():
        print(f"找不到源码目录：{SRC_DIR}", file=sys.stderr)
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sources = sorted(SRC_DIR.glob("*.py"))
    if not sources:
        print(f"{SRC_DIR} 里没有 .py 源文件", file=sys.stderr)
        return 1

    for src in sources:
        cells = parse_percent(src.read_text(encoding="utf-8"))
        if not cells:
            print(f"跳过（无有效 cell）：{src.name}")
            continue
        nb = to_ipynb(cells)
        dst = OUT_DIR / (src.stem + ".ipynb")
        dst.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        n_code = sum(1 for t, _ in cells if t == "code")
        print(f"生成 {dst.relative_to(ROOT)}  ({len(cells)} cells, {n_code} code)")

    print(f"\n完成，共 {len(sources)} 个 notebook。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
