#!/usr/bin/env python3
# encoding: utf-8
# coding style: pep8
# ====================================================
#   Copyright (C) 2026 ANQIN-X Project. All rights reserved.
#
#   Author        : An Qin
#   Email         : anqin.qin@gmail.com
#   File Name     : lfm_infer_only.py
#   Last Modified : 2026-07-01 15:55
#   Describe      : 
#
# ====================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
import math

# -------------------------- 全局配置 --------------------------
TOKENIZER_NAME = "gpt2"
CKPT_PATH = "./lfm_lm.pt"
MAX_GEN_TOKENS = 50
DIM = 512
NUM_LAYERS = 6
FILTER_ORDER = 64
NUM_SCALES = 3

# -------------------------- 模型结构（变量名100%对齐训练代码） --------------------------
class LongStateFilter(nn.Module):
    def __init__(self, dim: int, filter_order: int = 64, num_scales: int = 3):
        super().__init__()
        self.dim = dim
        self.num_scales = num_scales
        self.filter_order = filter_order

        self.filter_kernels = nn.ParameterList([
            nn.Parameter(torch.randn(filter_order, dim) / math.sqrt(filter_order))
            for _ in range(num_scales)
        ])
        self.scale_gate = nn.Linear(dim * num_scales, dim)
        self.norm = nn.LayerNorm(dim)

    def causal_filter(self, x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        K = self.filter_order
        x_t = x.transpose(1, 2)
        x_pad = F.pad(x_t, (K - 1, 0), mode="constant", value=0.0)
        k = kernel.transpose(0, 1).unsqueeze(1)
        out = F.conv1d(x_pad, k, padding=0, groups=D)
        out = out[..., :L]
        return out.transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm(x)
        scale_outputs = []
        for kernel in self.filter_kernels:
            s_out = self.causal_filter(x_norm, kernel)
            scale_outputs.append(s_out)
        concat = torch.cat(scale_outputs, dim=-1)
        mixed = self.scale_gate(concat)
        return x + mixed

class GatedMixerFFN(nn.Module):
    def __init__(self, dim: int, expand: float = 4.0):
        super().__init__()
        hidden = int(dim * expand)
        self.norm = nn.LayerNorm(dim)
        self.w1 = nn.Linear(dim, hidden * 2)
        self.w2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm(x)
        x_proj = self.w1(x_norm)
        gate, val = torch.chunk(x_proj, 2, dim=-1)
        swiglu = F.silu(gate) * val
        return x + self.w2(swiglu)

class LFMBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        # 关键：变量名必须是 filter_layer，和训练代码一致
        self.filter_layer = LongStateFilter(dim, FILTER_ORDER, NUM_SCALES)
        self.ffn = GatedMixerFFN(dim)

    def forward(self, x):
        x = self.filter_layer(x)
        x = self.ffn(x)
        return x

class LFMForLM(nn.Module):
    def __init__(self, vocab_size: int, dim: int, num_layers: int):
        super().__init__()
        # 变量名严格对齐训练代码
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.blocks = nn.Sequential(*[LFMBlock(dim) for _ in range(num_layers)])
        self.final_norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, input_ids):
        B, L = input_ids.shape
        x = self.token_emb(input_ids)
        x = self.blocks(x)
        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits

# -------------------------- 生成函数（最多50token） --------------------------
def generate(model, tokenizer, prompt, max_gen, device):
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    eos_token_id = tokenizer.eos_token_id

    with torch.no_grad():
        for _ in range(max_gen):
            logits = model(input_ids)
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            if next_token.item() == eos_token_id:
                break
    return tokenizer.decode(input_ids[0], skip_special_tokens=True)

# -------------------------- 主推理入口 --------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"推理设备: {device}")

    # 加载分词器（允许联网下载gpt2）
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    vocab_size = tokenizer.vocab_size

    # 初始化模型
    model = LFMForLM(vocab_size=vocab_size, dim=DIM, num_layers=NUM_LAYERS).to(device)
    # 加载训练保存的权重，key完全匹配不会报错
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device, weights_only=True))
    print("✅ 模型权重加载成功！最多续写50个token，输入 quit 退出\n")

    # 交互循环
    while True:
        prompt = input("Prompt > ")
        if prompt.strip().lower() == "quit":
            print("程序退出")
            break
        if not prompt.strip():
            print("输入不能为空\n")
            continue
        output_text = generate(model, tokenizer, prompt, MAX_GEN_TOKENS, device)
        print(f"Output: {output_text}\n")
