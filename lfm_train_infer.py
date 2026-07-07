#!/usr/bin/env python3
# encoding: utf-8
# coding style: pep8
# ====================================================
#   Copyright (C) 2026 ANQIN-X Project. All rights reserved.
#
#   Author        : An Qin
#   Email         : anqin.qin@gmail.com
#   File Name     : lfm_test2.py
#   Last Modified : 2026-07-01 15:15
#   Describe      : 
#
# ====================================================

import sys
# import os


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import math, os, tqdm

# ====================== 1. 分层长滤波器 LFM Core ======================
class LongStateFilter(nn.Module):
    def __init__(self, dim: int, filter_order: int, num_scales: int = 3):
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
        # 手动左填充实现因果
        x_pad = F.pad(x_t, (K - 1, 0), mode="constant", value=0.0)
        k = kernel.transpose(0, 1).unsqueeze(1)  # [D, 1, K]
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
    def __init__(self, dim: int, filter_order: int = 64, num_scales: int = 3, ffn_expand: float = 4.0):
        super().__init__()
        self.filter_layer = LongStateFilter(dim, filter_order, num_scales)
        self.ffn = GatedMixerFFN(dim, ffn_expand)

    def forward(self, x):
        x = self.filter_layer(x)
        x = self.ffn(x)
        return x

class LFMForLM(nn.Module):
    def __init__(
        self,
        vocab_size: int = 50257,
        dim: int = 512,
        num_layers: int = 6,
        filter_order: int = 64,
        num_scales: int = 3,
        ffn_expand: float = 4.0
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.blocks = nn.Sequential(*[
            LFMBlock(dim, filter_order, num_scales, ffn_expand)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, input_ids):
        B, L = input_ids.shape
        x = self.token_emb(input_ids)
        x = self.blocks(x)
        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits

# ====================== 2. 本地txt自定义数据集 ======================
class TextFileDataset(Dataset):
    def __init__(self, txt_path, tokenizer, seq_len):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        # 读取纯文本
        with open(txt_path, "r", encoding="utf-8") as f:
            self.lines = [line.strip() for line in f if len(line.strip()) > 0]

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        text = self.lines[idx]
        tokens = self.tokenizer(
            text,
            truncation=True,
            max_length=self.seq_len,
            padding="max_length",
            return_tensors="pt"
        )
        input_ids = tokens["input_ids"].squeeze(0)
        return input_ids

# ====================== 3. 损失、训练、验证函数 ======================
def calc_lm_loss(logits, input_ids):
    # 自回归：用前N-1预测后N-1
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=tokenizer.pad_token_id
    )
    return loss

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    pbar = tqdm.tqdm(loader, desc="Train")
    for ids in pbar:
        ids = ids.to(device)
        logits = model(ids)
        loss = calc_lm_loss(logits, ids)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix({"loss": loss.item()})
    return total_loss / len(loader)

def val_one_epoch(model, loader, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        pbar = tqdm.tqdm(loader, desc="Val")
        for ids in pbar:
            ids = ids.to(device)
            logits = model(ids)
            loss = calc_lm_loss(logits, ids)
            total_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})
    return total_loss / len(loader)

# ====================== 4. 文本生成推理函数 ======================
def generate_text(model, tokenizer, prompt, max_gen_len=128, device="cuda"):
    model.eval()
    tokens = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = tokens["input_ids"]
    with torch.no_grad():
        for _ in range(max_gen_len):
            logits = model(input_ids)
            # 取最后一个token预测
            next_logit = logits[:, -1, :]
            next_token = torch.argmax(next_logit, dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            if next_token.item() == tokenizer.eos_token_id:
                break
    out_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return out_text

# ====================== 5. 主执行入口 ======================
if __name__ == "__main__":
    # ---------------------- 配置参数 ----------------------
    SEQ_LEN = 512
    BATCH_SIZE = 4
    EPOCHS = 10
    LR = 6e-4
    TOKENIZER_NAME = "gpt2"
    # 本地文本文件路径
    TRAIN_TXT = "./train.txt"
    VAL_TXT = "./val.txt"
    CKPT_SAVE_PATH = "./lfm_lm.pt"

    # 设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Use device: {device}")

    # 分词器初始化
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 构建数据集与加载器
    train_ds = TextFileDataset(TRAIN_TXT, tokenizer, SEQ_LEN)
    val_ds = TextFileDataset(VAL_TXT, tokenizer, SEQ_LEN)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    # 初始化LFM模型
    model = LFMForLM(vocab_size=tokenizer.vocab_size, dim=512, num_layers=6).to(device)
    print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")

    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # ---------------------- 训练循环 ----------------------
    best_val_loss = float("inf")
    for ep in range(EPOCHS):
        print(f"\n===== Epoch {ep+1}/{EPOCHS} =====")
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = val_one_epoch(model, val_loader, device)
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # 保存最优模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), CKPT_SAVE_PATH)
            print(f"Saved best checkpoint to {CKPT_SAVE_PATH}")

    # ---------------------- 推理测试 ----------------------
    print("\n===== Inference Test =====")
    # 加载最优权重
    model.load_state_dict(torch.load(CKPT_SAVE_PATH, map_location=device))
    prompt = "Machine learning and long sequence models"
    gen_result = generate_text(model, tokenizer, prompt, max_gen_len=200, device=device)
    print("Prompt:", prompt)
    print("Generated Text:\n", gen_result)
