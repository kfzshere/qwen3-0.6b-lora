"""MMLU few-shot 评测（标准口径：纯文本 N-shot + 同学科示例 + logprob 打分）。
用于在和官方报告一致的口径下衡量真实水平/空间。
用法: python eval_mmlu_fewshot.py --model M --data test.parquet --dev dev.parquet --out o.json [--nshot 5]
"""
import argparse, json, time
import torch
import pyarrow.parquet as pq
from common import load_model

LETTERS = ["A", "B", "C", "D"]


def fmt_q(q, choices):
    s = q + "\n" + "\n".join(f"{LETTERS[i]}. {choices[i]}" for i in range(4))
    return s + "\nAnswer:"


def letter_ids(tok):
    ids = {}
    for L in LETTERS:
        cand = set()
        for form in (" " + L, L):
            e = tok.encode(form, add_special_tokens=False)
            if e:
                cand.add(e[0])
        ids[L] = list(cand)
    return ids


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--nshot", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=16)
    args = ap.parse_args()

    rows = pq.read_table(args.data).to_pylist()
    if args.limit:
        rows = rows[:args.limit]
    dev = pq.read_table(args.dev).to_pylist()
    by_sub = {}
    for d in dev:
        by_sub.setdefault(d["subject"], []).append(d)

    print(f"[mmlu-{args.nshot}shot] {len(rows)} 题, device={args.device}")
    t0 = time.time()
    model, tok = load_model(args.model, args.device)
    tok.padding_side = "left"
    lids = letter_ids(tok)

    prompts, golds = [], []
    for r in rows:
        sub = r["subject"]
        shots = by_sub.get(sub, [])[:args.nshot]
        header = f"The following are multiple choice questions (with answers) about {sub.replace('_', ' ')}.\n\n"
        ctx = header
        for s in shots:
            ctx += fmt_q(s["question"], s["choices"]) + f" {LETTERS[int(s['answer'])]}\n\n"
        ctx += fmt_q(r["question"], r["choices"])
        prompts.append(ctx)
        golds.append(int(r["answer"]))

    correct = 0
    for i in range(0, len(prompts), args.batch_size):
        batch = prompts[i:i + args.batch_size]
        enc = tok(batch, return_tensors="pt", padding=True,
                  truncation=True, max_length=3072).to(args.device)
        logits = model(**enc).logits[:, -1, :]
        logp = torch.log_softmax(logits.float(), dim=-1)
        for b in range(len(batch)):
            scores = [max(logp[b, tid].item() for tid in lids[L]) for L in LETTERS]
            pred = int(max(range(4), key=lambda k: scores[k]))
            correct += (pred == golds[i + b])
    acc = correct / len(prompts)
    res = {"task": f"mmlu_{args.nshot}shot", "n": len(prompts), "correct": correct,
           "accuracy": round(acc, 4), "sec": round(time.time() - t0, 1)}
    json.dump(res, open(args.out, "w"), ensure_ascii=False, indent=2)
    print(f"[mmlu-{args.nshot}shot] accuracy={acc:.4f} ({correct}/{len(prompts)}) -> {args.out}")


if __name__ == "__main__":
    main()
