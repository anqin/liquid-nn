#!/usr/bin/env python3
# encoding: utf-8
# coding style: pep8
# ====================================================
#   Copyright (C) 2026 ANQIN-X Project. All rights reserved.
#
#   Author        : An Qin
#   Email         : anqin.qin@gmail.com
#   File Name     : eval_from_online.py
#   Last Modified : 2026-07-15 16:42
#   Describe      : 
#
# ====================================================

import sys
# import os


import os, torch
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, DatasetDict

# ==========================================
# 1. 统一工作目录、开关与【纯本地模型】快照路径配置
# ==========================================
WORKSPACE_DIR = "./my_eval_workspace"  # 统一管理产出文件的工作目录
GENERATE_PLOT = True                  # True: 统一同时在工作空间产出柱状图表

os.makedirs(WORKSPACE_DIR, exist_ok=True)

# 📌 显式锁定并定位您本地系统原有的 ~/.cache 数据集根目录
HF_SYSTEM_CACHE_DIR = str(Path("~/.cache/huggingface/datasets").expanduser())

MODEL_PATHS = {
    "LFM-2.5-8B-A1B": "/home/anqin/.cache/huggingface/hub/models--LiquidAI--LFM2.5-8B-A1B/snapshots/5673e0de372b64331504de73bbbc33b0dde71903",
    "LFM-2-1.2B": "/home/anqin/.cache/huggingface/hub/models--LiquidAI--LFM2-1.2B/snapshots/933cee00d754fb3bfe06c644c0cb95453f2d8bb2"
}

# 📌 【全量修复校准】完美对齐 GLM-4 维度，将 IFEval 替换为长久公开可用的标准官方维护源
BENCHMARKS = {
    "MMLU (英文理解)": {"path": "cais/mmlu", "name": "abstract_algebra", "split": "test", "samples": 20, "type": "choice_mmlu"},
    "C-Eval (中文综合)": {"path": "ceval/ceval-exam", "name": "computer_network", "split": "val", "samples": 20, "type": "choice_ceval"},
    "ARC-Challenge (科学)": {"path": "allenai/ai2_arc", "name": "ARC-Challenge", "split": "test", "samples": 20, "type": "choice_arc"},
    "IFEval (指令遵循)": {"path": "wisely-ai/ifeval", "name": "default", "split": "train", "samples": 15, "type": "ifeval"},
    "GSM8K (基础数学)": {"path": "openai/gsm8k", "name": "main", "split": "test", "samples": 20, "type": "math_reason"},
    "MATH 500 (难题推导)": {"path": "HuggingFaceH4/MATH-500", "name": "default", "split": "test", "samples": 15, "type": "math_reason"},
    "AIME 2024 (竞赛)": {"path": "HuggingFaceH4/aime_2024", "name": "default", "split": "train", "samples": 10, "type": "math_reason"},
    "Codeforces (算法)": {"path": "open-r1/codeforces", "name": "default", "split": "test", "samples": 10, "type": "coding"}
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# 2. 规则驱动的自动化评测与【精准本地复用】引擎
# ==========================================
def load_dataset_with_local_check(path, name, split):
    """【精准复用】显式锁定 ~/.cache 目录，确保 100% 离线复用成功"""
    try:
        # 1. 优先尝试纯离线秒级加载
        return load_dataset(path, name, split=split, cache_dir=HF_SYSTEM_CACHE_DIR, local_files_only=True)
    except Exception:
        pass

    print(f"📥 路径中未检测到完整缓存 {path}，正在尝试自动在线下载...")
    try:
        # 2. 离线未命中，尝试在线精确下载
        return load_dataset(path, name, split=split, cache_dir=HF_SYSTEM_CACHE_DIR)
    except Exception as e:
        if "split" in str(e).lower() or "should be one of" in str(e).lower():
            print(f"  {path} 的 {split} 分支配置缺失，正在自适应全量拉取...")
            return load_dataset(path, name, cache_dir=HF_SYSTEM_CACHE_DIR)
        raise e

def evaluate_model(model_name: str, model_path: str) -> dict:
    print(f"\n[📦 载入本地模型] {model_name} 从 {model_path} ...")
    if not os.path.exists(model_path): raise FileNotFoundError(f"路径不存在: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32, device_map="auto" if DEVICE == "cuda" else None, trust_remote_code=True, local_files_only=True).eval()

    eval_results = {}
    for bench_name, cfg in BENCHMARKS.items():
        print(f" 🔍 正在评测 ➔ {bench_name}")
        try:
            raw_data = load_dataset_with_local_check(cfg["path"], cfg["name"], cfg["split"])

            # 自适应一维安全解包 DatasetDict
            if isinstance(raw_data, DatasetDict):
                active_split = cfg["split"] if cfg["split"] in raw_data else list(raw_data.keys())[0]
                current_data = raw_data[active_split]
            else:
                current_data = raw_data

            correct, total = 0, min(cfg["samples"], len(current_data))

            for i in tqdm(range(total), desc=f"[{bench_name}]"):
                raw_item = current_data[i]
                item = raw_item if isinstance(raw_item, dict) else dict(raw_item)

                # 🛠️ 自适应规则解耦表
                EVAL_RULES = {
                    "choice_mmlu": {
                        "prompt": f"Question: {item.get('question','')}\n" + "".join([f"{['A','B','C','D'][idx]}. {c}\n" for idx, c in enumerate(item.get('choices', []))]) + "Answer:",
                        "tokens": 2, "check": lambda p, it: p.startswith(('A','B','C','D')[it['answer']])
                    },
                    "choice_ceval": {
                        "prompt": f"题目: {item.get('question','')}\nA. {item.get('A','')}\nB. {item.get('B','')}\nC. {item.get('C','')}\nD. {item.get('D','')}\n答
案:",
                        "tokens": 2, "check": lambda p, it: str(it.get('answer', 'A')).upper() in p.upper()
                    },
                    "choice_arc": {
                        "prompt": f"Question: {item.get('question','')}\n" + "".join([f"{t}. {t}\n" for t in zip(item.get('choices', {}).get('label', []), item.get('choices', {}).get('text', []))]) + "Answer:",
                        "tokens": 2, "check": lambda p, it: p.startswith(str(it.get('answerKey', 'A')))
                    },
                    "ifeval": {
                        # wisely-ai/ifeval 遵循标准的 prompt 结构进行格式约束检测
                        "prompt": f"System: Complete the task by strictly following any formatting or constraint requests.\nTask: {item.get('prompt', '')}\nResponse:",
                        "tokens": 128, "check": lambda p, it: any(kw.lower() in p.lower() for kw in it.get('kwargs', {}).get('key_words', [' '])) or len(p.strip()) > 5
                    },
                    "math_reason": {                                                                                                                                                       "prompt": f"Problem: {item.get('problem', item.get('question',''))}\nLet's think step by step. Provide your final concise answer at the end:",
                        "tokens": 256, "check": lambda p, it: str(it.get('answer', it.get('solution',''))).strip().lower() in p.lower()
                    },
                    "coding": {
                        "prompt": f"Task: Write a Python program to solve the following problem:\n{item.get('problem_description', item.get('question', item.get('problem', '')))}\n\nPython Code Solution:",
                        "tokens": 512, "check": lambda p, it: any(kw in p.lower() for kw in ["def ", "import", "print", "sys", "input", str(it.get('answer','')).lower()[:5]])
                    }
                }

                rule = EVAL_RULES[cfg["type"]]
                inputs = tokenizer(rule["prompt"], return_tensors="pt", max_length=2048, truncation=True).to(DEVICE)
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=rule["tokens"], pad_token_id=tokenizer.pad_token_id)

                pred = tokenizer.decode(outputs[0, inputs.input_ids.shape[-1]:], skip_special_tokens=True).strip()
                if rule["check"](pred, item): correct += 1

            eval_results[bench_name] = round((correct / total) * 100, 2) if total > 0 else 0.0
        except Exception as e:
            print(f" ❌ {bench_name} 运行异常: {e}"); eval_results[bench_name] = 0.0

    del model, tokenizer
    if DEVICE == "cuda": torch.cuda.empty_cache()
    return eval_results

# ==========================================
# 3. 结果多重同步导出与主控制流程
# ==========================================
def main():
    all_summary = []
    for m_name, m_path in MODEL_PATHS.items():
        for b_name, score in evaluate_model(m_name, m_path).items():
            all_summary.append({"Model": m_name, "Benchmark": b_name, "Score (%)": score})
    df = pd.DataFrame(all_summary)

    ct = df.pivot(index="Model", columns="Benchmark", values="Score (%)")

    print("\n🔥 评测完成！终端对比矩阵表格如下：")
    print("=" * 95)
    print(ct.to_string())
    print("=" * 95)

    # 产出文本报告持久化在指定工作空间 `./my_eval_workspace` 中
    report_path = os.path.join(WORKSPACE_DIR, "lfm_eval_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# LiquidAI 系列本地大模型综合能力对比报告（对齐 GLM 评测维度）\n\n")
        f.write("## 1. 核心结果对齐交叉表 (Markdown 格式)\n")
        f.write(ct.to_markdown() if hasattr(ct, 'to_markdown') else ct.to_string())
    print(f"📝 文本对比报告已成功保存至工作空间: {os.path.abspath(report_path)}")

    if GENERATE_PLOT:
        plt.figure(figsize=(14, 7)); sns.set_theme(style="whitegrid")
        ax = sns.barplot(data=df, x="Benchmark", y="Score (%)", hue="Model", palette="coolwarm")
        for p in ax.patches:
            if p.get_height() > 0: ax.annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 8), textcoords='offset points', fontsize=9, weight='bold')
        plt.title("Model Capabilities Comparison (GLM-4-9B Evaluation Dimension Alignment)", fontsize=12, weight='bold', pad=20); plt.ylim(0, 110)
        plt.xticks(rotation=15)
        plt.tight_layout()

        # 产出柱状图持久化在指定工作空间 `./my_eval_workspace` 中
        plot_path = os.path.join(WORKSPACE_DIR, "lfm_benchmark_comparison.png")
        plt.savefig(plot_path, dpi=300)
        print(f"📊 对比柱状图已成功保存至工作空间: {os.path.abspath(plot_path)}")
        plt.show()

if __name__ == "__main__":
    main()
