"""拒绝采样造数学训练数据：用基座模型自己解 GSM8K，只保留答对的（原生风格+正确）。
用法: python rejection_sample_math.py --model <基座> --data <gsm8k_train.parquet> --out <math_rs.json> [--limit N]
"""
import argparse, json, sys, os, re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "eval"))
import pyarrow.parquet as pq
from common import load_model, chat_generate, extract_gold_gsm8k, extract_pred_number

HINT = "\nPlease reason step by step, and put your final answer after '#### '."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    args = ap.parse_args()

    rows = pq.read_table(args.data).to_pylist()
    if args.limit:
        rows = rows[:args.limit]
    questions = [r["question"] + HINT for r in rows]
    golds = [extract_gold_gsm8k(r["answer"]) for r in rows]

    print(f"[rs] 生成 {len(questions)} 题(非thinking)...")
    model, tok = load_model(args.model, args.device)
    outs = chat_generate(model, tok, questions, args.device,
                         max_new_tokens=args.max_new_tokens,
                         batch_size=args.batch_size, enable_thinking=False)
    kept = []
    for i, out in enumerate(outs):
        pred = extract_pred_number(out)
        if pred is not None and pred == golds[i]:
            sol = out.strip()
            # 确保结尾有 #### 答案，便于评测抽取
            if "####" not in sol:
                sol += f"\n#### {golds[i]}"
            kept.append({"instruction": rows[i]["question"] + HINT,
                         "input": "", "output": sol, "task": "math"})
    json.dump(kept, open(args.out, "w"), ensure_ascii=False, indent=1)
    print(f"[rs] 保留正确 {len(kept)}/{len(questions)} "
          f"({len(kept)/len(questions)*100:.1f}%) -> {args.out}")


if __name__ == "__main__":
    main()
