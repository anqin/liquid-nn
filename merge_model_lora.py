#!/usr/bin/env python3
# encoding: utf-8
# coding style: pep8
# ====================================================
#   Copyright (C) 2026 ANQIN-X Project. All rights reserved.
#
#   Author        : An Qin
#   Email         : anqin.qin@gmail.com
#   File Name     : merge_model_lora.py
#   Last Modified : 2026-07-10 18:03
#   Describe      : 
#
# ====================================================

import sys
# import os


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 1. 严格使用你报错日志中的【真实绝对路径】
base_model_path = "/home/anqin/.cache/huggingface/hub/models--LiquidAI--LFM2.5-350M/snapshots/b9d6e4e2d75f440b12a2b4d731c808004ecbbd89/"
adapter_path = "/home/anqin/leap-finetune/outputs/my_sft_project_anqin/LFM2.5-350M-sft-smoltalk-1000-lr2em05-w0p2-lora_a-20260710_153135/latest/"
output_path = "./outputs/my_complete_hf_model" # 转换成功后的完整 HF 格式目录

print("⚡ 正在从本地缓存载入 LFM2.5 基座模型与 Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16,  # 保持 2.5 推荐的精度
    device_map="cpu",            # 优先在内存中进行融合，防止 GPU 显存碎掉
    trust_remote_code=True
)

print("⚡ 正在挂载你的最新微调 LoRA 权重...")
model = PeftModel.from_pretrained(base_model, adapter_path)

print("⚡ 正在执行矩阵硬合并 (Merge & Unload)...")
merged_model = model.merge_and_unload()

print("⚡ 正在保存补全后的完整模型（此时将自动生成 config.json!）...")
merged_model.save_pretrained(output_path, safe_serialization=True)
tokenizer.save_pretrained(output_path)

print(f"✅ 完美转换！完整的 Hugging Face 模型已保存在: {output_path}")
