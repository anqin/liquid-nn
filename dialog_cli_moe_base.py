#!/usr/bin/env python3
# encoding: utf-8
# coding style: pep8
# ====================================================
#   Copyright (C) 2026 ANQIN-X Project. All rights reserved.
#
#   Author        : An Qin
#   Email         : anqin.qin@gmail.com
#   File Name     : dialog_cli_moe_base.py
#   Last Modified : 2026-07-15 10:35
#   Describe      : 
#
# ====================================================

import sys
# import os


import os
import sys
import io
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

# 强行将系统的输入输出流指定为 utf-8 编码，防止中文输入崩溃
sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8', errors='replace')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# === 确切的本地 LFM2.5-8B-A1B 模型绝对路径 ===
LOCAL_MODEL_PATH = "/home/anqin/.cache/huggingface/hub/models--LiquidAI--LFM2.5-8B-A1B/snapshots/5673e0de372b64331504de73bbbc33b0dde71903"

def main():
    print("⏳ 正在以全精度模式加载 LFM2.5-8B 旗舰模型...")
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        LOCAL_MODEL_PATH,
        dtype=torch.bfloat16,   # 全精度不妥协，完美释放 8B 模型的全部智商
        device_map="auto",      # 自动利用本地所有的 GPU/CPU 硬件能力
        trust_remote_code=True
    )
    print("✅ LFM2.5-8B 满血全能力解锁成功！")

    # 标准的 Hugging Face 消息池结构，天然支持多轮记忆隔离
    messages = []

    print("\n🤖 欢迎来到 LFM2.5-8B 旗舰控制台！输入 'exit' 退出对话。\n" + "="*50)

    while True:
        user_input = input("🧑 User: ").strip()
        if not user_input or user_input.lower() == 'exit':
            print("👋 再见！")
            break

        # 1. 存入当前用户提问
        messages.append({"role": "user", "content": user_input})

        # 2. 调用模型内置的结构化聊天模板
        # 这里返回的是字典 {'input_ids': ..., 'attention_mask': ...}
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,  # 引导模型自动生成正确的回复头部
            return_tensors="pt"
        ).to(model.device)

        # 3. 改用官方原生的流式打印器
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        print("🤖 Assistant: ", end="", flush=True)

        # 4. 【核心修复】使用 **inputs 解包字典，并将 input_ids 的第 1 维长度作为切片基准
        input_length = inputs['input_ids'].shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,                # 加两个星号解包字典，解决 KeyError 报错
                streamer=streamer,       # 原生流式发射
                max_new_tokens=1024,     # 充分发挥大模型的长文本推理能力
                do_sample=True,
                temperature=0.7,         # 兼顾逻辑与思维开阔度
                pad_token_id=tokenizer.eos_token_id
            )

        # 5. 从输出中解码出纯净的助手回答，存入下一轮记忆
        response_tokens = outputs[0][input_length:]
        model_response = tokenizer.decode(response_tokens, skip_special_tokens=True).strip()

        messages.append({"role": "assistant", "content": model_response})
        print("-" * 50 + "\n")

if __name__ == "__main__":
    main()
