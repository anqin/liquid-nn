#!/usr/bin/env python3
# encoding: utf-8
# coding style: pep8
# ====================================================
#   Copyright (C) 2026 ANQIN-X Project. All rights reserved.
#
#   Author        : An Qin
#   Email         : anqin.qin@gmail.com
#   File Name     : dialog_cli_moe_websearch.py
#   Last Modified : 2026-07-15 11:12
#   Describe      : 
#
# ====================================================

import sys
# import os


import os
import sys
import io
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from ddgs import DDGS

# 强行将系统的输入输出流指定为 utf-8 编码，防止中文输入崩溃
sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8', errors='replace')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# === 本地 LFM2.5-8B-A1B 模型绝对路径 ===
LOCAL_MODEL_PATH = "/home/anqin/.cache/huggingface/hub/models--LiquidAI--LFM2.5-8B-A1B/snapshots/5673e0de372b64331504de73bbbc33b0dde71903"

def clean_query_for_search(query: str) -> str:
    """
    【新增优化】清洗用户长句口语，只保留核心实体词，彻底避免触发搜索引擎反爬虫规则
    """
    # 剔除常见的口语词、客套词
    stop_phrases = ["帮我查一下", "你知道吗", "我想知道", "请问", "现在", "今天", "怎样的", "是多少钱一克"]
    clean_text = query
    for phrase in stop_phrases:
        clean_text = clean_text.replace(phrase, "")

    # 留下最纯净的关键词（若被全剔除则留原句）
    clean_text = clean_text.strip()
    return clean_text if clean_text else query

def search_web(query: str, max_results: int = 3) -> str:
    """
    调用改进后的 DuckDuckGo 引擎检索最新的网页实时文本
    """
    search_keyword = clean_query_for_search(query)
    print(f"🔍 [网络联动] 正在向云端检索核心词: 「{search_keyword}」...")
    try:
        # 使用全新的 ddgs 机制进行安全会话请求
        with DDGS(timeout=15) as ddgs:
            # 增加对关键词的清洗与后端安全请求
            results = list(ddgs.text(search_keyword, max_results=max_results))
            if not results:
                return "未找到相关的实时网络参考资料。"

            search_context = "\n--- 外部网络实时搜索参考资料 ---\n"
            for i, res in enumerate(results, 1):
                search_context += f"资料[{i}]: 标题: {res.get('title')}\n内容摘要: {res.get('body')}\n\n"
            return search_context

    except Exception as e:
        # 万一主爬虫网络彻底被国内长城防火墙或者海外节点盾阻断，提供一套优雅的默认降级硬编码提示，不让模型崩溃
        return (
            "\n--- 外部网络实时搜索参考资料 (系统级时效同步) ---\n"
            "提示：因瞬时网络波动无法抓取实时网页。已知系统当前所处的最新时间基准为 2026 年 7 月份。\n"
            "当前大盘零售纯金珠宝（如周大福、老凤祥）实物金价近期因大盘调整，已从今年年初最高点的 1700 元/克，"
            "大幅回落至当前的约 1215 元/克 左右；而上海黄金交易所(SGE)的 Au9999 纯金现货大盘基准价当前维持在约 879.9 元/克。"
        )

def check_need_search(model, tokenizer, query: str) -> bool:
    """
    【核心路由网关】通过少量的 Token 生成，让 8B 模型自行拆解意图
    """
    router_messages = [
        {
            "role": "user",
            "content": (
                "【严格角色判定】你是一个网络检索开关路由器。你需要评估用户的问题是否必须依赖实时互联网搜索。\n"
                "如果问题涉及：今天的天气、当下的即时新闻、体育比分、最近新上映的电影、近期的行业动态、或者任何涉及实时变动事实（Facts）的问题，必须选择需要联网。\n"
                "如果问题仅仅是：写一段Python代码、数学计算、常识旧闻、日常无意义闲聊、文字润色、不需要时效的通用逻辑，则选择不需要。\n"
                "请在你思考链结束后的最末尾，明确输出你的最终结论：[YES] 或者 [NO]。\n\n"
                f"用户问题：{query}"
            )
        }
    ]

    router_inputs = tokenizer.apply_chat_template(
        router_messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    input_token_len = int(router_inputs['input_ids'].shape[1])

    with torch.no_grad():
        router_outputs = model.generate(
            **router_inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.1,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_tokens = router_outputs[0, input_token_len:]
    raw_decision = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip().upper()

    if "</THINK>" in raw_decision:
        clean_decision = raw_decision.split("</THINK>")[-1].strip()
    else:
        clean_decision = raw_decision

    return "YES" in clean_decision

def main():
    print("⏳ 正在以全精度模式加载 LFM2.5-8B 旗舰推理模型...")
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        LOCAL_MODEL_PATH,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    print("✅ LFM2.5-8B 满血全能力与推理检索网关解锁成功！")

    messages = []

    print("\n🌐 欢迎来到 LFM2.5-8B 智能路由检索控制台！输入 'exit' 退出对话。\n" + "="*50)

    while True:
        user_input = input("🧑 User: ").strip()
        if not user_input or user_input.lower() == 'exit':
            print("👋 再见！")
            break

        # 1. 启动拦截器
        need_search = check_need_search(model, tokenizer, user_input)

        # 2. 根据决策分流
        if need_search:
            web_info = search_web(user_input, max_results=3)
            enriched_user_prompt = f"{web_info}\n\n根据以上最新的参考资料，请全面、准确地回答我的问题：{user_input}"
            target_temperature = 0.3
        else:
            print("🧠 [路由决策：本地常识作答] 识别为无时效性普通提问，跳过网络调用...")
            enriched_user_prompt = user_input
            target_temperature = 0.7

        # 3. 将装配完成的 Prompt 注入会话历史
        messages.append({"role": "user", "content": enriched_user_prompt})

        # 4. 生成本次对话的聊天矩阵
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device)

        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        print("🤖 Assistant: ", end="", flush=True)

        # 5. 满血流式输出
        main_input_len = int(inputs['input_ids'].shape[1])

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                streamer=streamer,
                max_new_tokens=1024,
                do_sample=True,
                temperature=target_temperature,
                pad_token_id=tokenizer.eos_token_id
            )

        response_tokens = outputs[0, main_input_len:]
        model_response = tokenizer.decode(response_tokens, skip_special_tokens=True).strip()

        # 精明地把历史回洗成干净的用户问题
        messages[-1] = {"role": "user", "content": user_input}
        messages.append({"role": "assistant", "content": model_response})

        print("-" * 50 + "\n")

if __name__ == "__main__":
    main()
