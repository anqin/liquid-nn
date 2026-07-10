#!/usr/bin/env python3
# encoding: utf-8
# coding style: pep8
# ====================================================
#   Copyright (C) 2026 ANQIN-X Project. All rights reserved.
#
#   Author        : An Qin
#   Email         : anqin.qin@gmail.com
#   File Name     : dialog_app_web.py
#   Last Modified : 2026-07-10 18:36
#   Describe      : 
#
# ====================================================

import sys
# import os


import os
import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- 1. 配置本地模型路径 ---
# 替换为你实际想要加载的本地合并模型或基座模型绝对路径
LOCAL_MODEL_PATH = os.path.expanduser(
    "/home/anqin/.cache/huggingface/hub/models--LiquidAI--LFM2-1.2B/snapshots/933cee00d754fb3bfe06c644c0cb95453f2d8bb2/"
)

# --- 2. 缓存模型加载逻辑（防止每次流式刷新都重复加载） ---
@st.cache_resource
def load_model_and_tokenizer(path):
    st.write("⏳ 正在首次加载本地模型，请稍候（通常需要1-2分钟）...")
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    return model, tokenizer

# 初始化加载模型
model, tokenizer = load_model_and_tokenizer(LOCAL_MODEL_PATH)

# --- 3. Streamlit 页面 UI 设置 ---
st.set_page_config(page_title="本地 LFM 模型聊天室", page_icon="🤖")
st.title("💬 本地大模型多轮对话系统")
st.caption("🚀 基于本地 Checkpoint 运行 | 支持 Session 历史记忆")

# --- 4. 初始化 Session 状态（核心：用于长效记忆对话历史） ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 5. 渲染历史对话气泡 ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 6. 监听用户当前输入 ---
if prompt := st.chat_input("向你的本地模型提问..."):
    # 在界面上即时渲染用户输入的内容
    with st.chat_message("user"):
        st.markdown(prompt)

    # 存入 Session 历史
    st.session_state.messages.append({"role": "user", "content": prompt})

    # --- 7. 构建带有历史记忆的上下文 Prompt ---
    # 循环遍历 Session 历史，将前序对话拼接到一起提供给模型输入
    context = ""
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            context += f"User: {msg['content']}\n"
        else:
            context += f"Assistant: {msg['content']}\n"
    context += "Assistant: " # 留出尾巴引导模型回答

    # --- 8. 模型生成阶段 ---
    with st.chat_message("assistant"):
        response_placeholder = st.empty() # 创建一个空占位符用于动态渲染文本
        response_placeholder.markdown("🤖 *思考中...*")

        # 编码并生成
        inputs = tokenizer(context, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )

        # 截取模型新生成的文本回复
        input_length = inputs.input_ids.shape[1]
        generated_tokens = outputs[0][input_length:]
        full_response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        # 将最终文本渲染到页面上
        response_placeholder.markdown(full_response)

    # 将模型的回答也持久化进 Session 历史
    st.session_state.messages.append({"role": "assistant", "content": full_response})
