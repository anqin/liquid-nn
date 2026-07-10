#!/usr/bin/env python3
# encoding: utf-8
# coding style: pep8
# ====================================================
#   Copyright (C) 2026 ANQIN-X Project. All rights reserved.
#
#   Author        : An Qin
#   Email         : anqin.qin@gmail.com
#   File Name     : merged_model_run.py
#   Last Modified : 2026-07-10 18:13
#   Describe      : 
#
# ====================================================

import sys
# import os


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 注：model_path不可以是 “~”
model_path = "/home/anqin/.cache/huggingface/hub/models--LiquidAI--LFM2-1.2B/snapshots/933cee00d754fb3bfe06c644c0cb95453f2d8bb2/" 
# model_path = "./outputs/my_complete_hf_model"

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

# 测试你的微调效果
prompt = "你好，请问你是谁？"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=50)

print("\n--- 模型微调后回复 ---")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
