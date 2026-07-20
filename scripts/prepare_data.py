"""把三类原始数据转成 LLaMA-Factory alpaca 格式并按比例混合。
用法: python prepare_data.py --base <qwen-demo路径> [--n_safety 4000 --n_math 3000 --n_general 3000]
输出: <base>/datasets/lf_data/mixed_sft_v1.json + dataset_info.json
"""
import argparse, json, os, re, random
import pyarrow.parquet as pq

MATH_HINT = "\nPlease reason step by step, and put your final answer after '#### '."


def clean_gsm8k(ans):
    """去掉 <<...>> 计算器标记，保留自然语言推理 + '#### N'。"""
    return re.sub(r"<<[^>]*>>", "", ans).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--n_safety", type=int, default=4000)
    ap.add_argument("--n_math", type=int, default=3000)
    ap.add_argument("--n_general", type=int, default=3000)
    ap.add_argument("--out_name", default="mixed_sft_v1")
    ap.add_argument("--math_rs", default="", help="拒绝采样数学数据(json)，给了就用它替代原始GSM8K")
    args = ap.parse_args()
    random.seed(42)
    ds = os.path.join(args.base, "datasets")
    out_dir = os.path.join(ds, "lf_data")
    os.makedirs(out_dir, exist_ok=True)

    # 1. 安全：有害 prompt -> 拒答
    aegis = json.load(open(f"{ds}/safety/aegis_refusals_train.json"))
    aegis = [d for d in aegis if d.get("prompt_label") == "unsafe"
             and d.get("prompt") and d.get("response")]
    random.shuffle(aegis)
    safety = [{"instruction": d["prompt"], "input": "", "output": d["response"],
               "task": "safety"} for d in aegis[:args.n_safety]]

    # 2. 数学：GSM8K 短 CoT，或拒绝采样的原生风格正确解答
    if args.math_rs:
        rs = json.load(open(args.math_rs))
        random.shuffle(rs)
        math = [{"instruction": d["instruction"], "input": "",
                 "output": d["output"], "task": "math"} for d in rs[:args.n_math]]
    else:
        rows = pq.read_table(f"{ds}/math/gsm8k_train.parquet").to_pylist()
        random.shuffle(rows)
        math = [{"instruction": r["question"] + MATH_HINT, "input": "",
                 "output": clean_gsm8k(r["answer"]), "task": "math"}
                for r in rows[:args.n_math]]

    # 3. 通用：Alpaca replay
    alp = json.load(open(f"{ds}/general/alpaca_cleaned_3000.json"))
    random.shuffle(alp)
    general = [{"instruction": d["instruction"], "input": d.get("input", ""),
                "output": d["output"], "task": "general"} for d in alp[:args.n_general]]

    mixed = safety + math + general
    random.shuffle(mixed)
    out_file = f"{out_dir}/{args.out_name}.json"
    json.dump(mixed, open(out_file, "w"), ensure_ascii=False, indent=1)

    # dataset_info.json（LLaMA-Factory 注册）
    info_path = f"{out_dir}/dataset_info.json"
    info = json.load(open(info_path)) if os.path.exists(info_path) else {}
    info[args.out_name] = {"file_name": f"{args.out_name}.json"}
    json.dump(info, open(info_path, "w"), ensure_ascii=False, indent=2)

    from collections import Counter
    print(f"混合完成: {len(mixed)} 条 -> {out_file}")
    print("配比:", dict(Counter(x["task"] for x in mixed)))
    print("dataset_info:", info_path)
    print("样本(math):", next(x for x in mixed if x["task"] == "math")["output"][:120])


if __name__ == "__main__":
    main()
