#!/usr/bin/env python3
# encoding: utf-8
# coding style: pep8
# ====================================================
#   Copyright (C) 2026 ANQIN-X Project. All rights reserved.
#
#   Author        : An Qin
#   Email         : anqin.qin@gmail.com
#   File Name     : LFM_test.py
#   Last Modified : 2026-07-01 15:08
#   Describe      : 
#
# ====================================================

import sys
# import os


import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --------------------------
# 1. 分层长状态滤波器 Hierarchical Long Filter（已修复conv1d padding报错）
# --------------------------
class LongStateFilter(nn.Module):
    def __init__(self, dim: int, filter_order: int, num_scales: int = 3):
        super().__init__()
        self.dim = dim
        self.num_scales = num_scales
        self.filter_order = filter_order

        # 多尺度分层滤波器参数 (论文分层设计)
        self.filter_kernels = nn.ParameterList([
            nn.Parameter(torch.randn(filter_order, dim) / math.sqrt(filter_order))
            for _ in range(num_scales)
        ])
        # 尺度门控，融合不同尺度滤波输出
        self.scale_gate = nn.Linear(dim * num_scales, dim)
        self.norm = nn.LayerNorm(dim)

    def causal_filter(self, x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        """因果一维卷积滤波，修复padding错误
        x: [B, L, D]
        kernel: [K, D]
        return: [B, L, D]
        """
        B, L, D = x.shape
        K = self.filter_order
        # [B,L,D] -> [B,D,L]
        x_t = x.transpose(1, 2)

        # 手动左填充 K-1 个0，右侧不填充，实现因果
        # F.pad格式：(左, 右) 作用于最后一维
        x_pad = F.pad(x_t, (K - 1, 0), mode="constant", value=0.0)

        # 卷积核 [K,D] -> [D, 1, K]，分组卷积每组一个通道
        k = kernel.transpose(0, 1).unsqueeze(1)  # [D, 1, K]

        # conv1d padding=0，输入已手动填充
        out = F.conv1d(x_pad, k, padding=0, groups=D)
        # 截取前L长度，消除填充带来的多余长度
        out = out[..., :L]
        return out.transpose(1, 2)  # [B,L,D]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        x_norm = self.norm(x)
        # 多尺度分层滤波
        scale_outputs = []
        for kernel in self.filter_kernels:
            s_out = self.causal_filter(x_norm, kernel)
            scale_outputs.append(s_out)
        # 拼接多尺度特征 + 门控融合
        concat = torch.cat(scale_outputs, dim=-1)  # [B,L,D*num_scales]
        mixed = self.scale_gate(concat)
        # 残差连接
        return x + mixed

# --------------------------
# 2. Gated Mixer FFN (SwiGLU 论文标准)
# --------------------------
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

# --------------------------
# 3. LFM Block 完整层 (论文标准块结构)
# Filter -> FFN，两层残差
# --------------------------
class LFMBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        filter_order: int = 64,
        num_scales: int = 3,
        ffn_expand: float = 4.0
    ):
        super().__init__()
        self.filter_layer = LongStateFilter(dim, filter_order, num_scales)
        self.ffn = GatedMixerFFN(dim, ffn_expand)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.filter_layer(x)
        x = self.ffn(x)
        return x

# --------------------------
# 4. LFM 完整语言建模模型 (Causal LM)
# --------------------------
class LFMForLM(nn.Module):
    def __init__(
        self,
        vocab_size: int = 50257,
        dim: int = 512,
        num_layers: int = 6,
        filter_order: int = 64,
        num_scales: int = 3,
        ffn_expand: float = 4.0,
        max_seq_len: int = 8192
    ):
        super().__init__()
        self.dim = dim
        # Embedding
        self.token_emb = nn.Embedding(vocab_size, dim)
        # 无显式位置编码：长滤波器天然捕获时序位置依赖（论文核心结论）
        # LFM 堆叠层
        self.blocks = nn.Sequential(*[
            LFMBlock(dim, filter_order, num_scales, ffn_expand)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(dim)
        # LM Head
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """
        input_ids: [B, L] token indices
        return: [B, L, vocab_size] logits
        """
        B, L = input_ids.shape
        x = self.token_emb(input_ids)  # [B, L, D]
        x = self.blocks(x)
        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits

# --------------------------
# 自回归LM损失函数
# --------------------------
def lm_loss(logits, labels):
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    )
    return loss

# --------------------------
# 测试代码：验证模型前向传播（可直接运行无报错）
# --------------------------
if __name__ == "__main__":
    # 超参匹配论文小基线配置
    VOCAB = 50257
    DIM = 512
    LAYERS = 6
    SEQ_LEN = 1024
    BATCH = 2

    # 初始化LFM模型
    model = LFMForLM(
        vocab_size=VOCAB,
        dim=DIM,
        num_layers=LAYERS,
        filter_order=64,
        num_scales=3
    )
    # 模拟输入token
    dummy_input = torch.randint(0, VOCAB, (BATCH, SEQ_LEN))
    # 前向传播
    logits = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Model total params: {sum(p.numel() for p in model.parameters()):,}")

    # 测试loss
    loss_val = lm_loss(logits, dummy_input)
    print(f"Test loss value: {loss_val.item():.4f}")
