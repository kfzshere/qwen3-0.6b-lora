"""数学基线评测：GSM8K test 准确率。"""
import argparse, json, time
import pyarrow.parquet as pq
from common import load_model, chat_generate, extract_gold_gsm8k, extract_pred_number

INSTR = "\nPlease reason step by step, and put your final answer after '#### '."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--thinking", action="store_true")
    args = ap.parse_args()

    t = pq.read_table(args.data).to_pylist()
    if args.limit:
        t = t[:args.limit]
    questions = [r["question"] + INSTR for r in t]
    golds = [extract_gold_gsm8k(r["answer"]) for r in t]

    print(f"[math] {len(questions)} 题, device={args.device}, thinking={args.thinking}")
    t0 = time.time()
    model, tok = load_model(args.model, args.device)
    preds_txt = chat_generate(model, tok, questions, args.device,
                              max_new_tokens=args.max_new_tokens,
                              batch_size=args.batch_size,
                              enable_thinking=args.thinking)
    correct = 0
    samples = []
    for i, out in enumerate(preds_txt):
        pred = extract_pred_number(out)
        ok = pred is not None and pred == golds[i]
        correct += ok
        if i < 5:
            samples.append({"q": t[i]["question"][:80], "gold": golds[i],
                            "pred": pred, "ok": ok, "out": out[:200]})
    acc = correct / len(questions)
    res = {"task": "math_gsm8k", "n": len(questions), "correct": correct,
           "accuracy": round(acc, 4), "sec": round(time.time() - t0, 1),
           "thinking": args.thinking, "samples": samples}
    json.dump(res, open(args.out, "w"), ensure_ascii=False, indent=2)
    print(f"[math] accuracy={acc:.4f} ({correct}/{len(questions)}) "
          f"用时 {res['sec']}s -> {args.out}")


if __name__ == "__main__":
    main()
