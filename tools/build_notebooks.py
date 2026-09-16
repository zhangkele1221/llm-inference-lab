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
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# MiniGPT 只有 2700 万参数，用 float16 跑在 GPU 上；CPU 上 float16 很慢，用 float32
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32


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
