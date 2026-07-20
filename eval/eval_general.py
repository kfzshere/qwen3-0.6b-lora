"""通用基线评测：MMLU 准确率（用 A/B/C/D 首 token 对数概率打分，适合小模型）。"""
import argparse, json, time
import torch
import pyarrow.parquet as pq
from common import load_model

LETTERS = ["A", "B", "C", "D"]


def letter_token_ids(tok):
    """每个字母的候选 token id（含带前导空格的变体），取 logit 时按候选取 max。"""
    ids = {}
    for L in LETTERS:
        cand = set()
        for form in (L, " " + L):
            enc = tok.encode(form, add_special_tokens=False)
            if enc:
                cand.add(enc[0])
        ids[L] = list(cand)
    return ids


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()

    rows = pq.read_table(args.data).to_pylist()
    if args.limit:
        rows = rows[:args.limit]

    print(f"[general] MMLU {len(rows)} 题, device={args.device}")
    t0 = time.time()
    model, tok = load_model(args.model, args.device)
    lids = letter_token_ids(tok)

    prompts = []
    golds = []
    for r in rows:
        ch = r["choices"]
        q = r["question"] + "\n" + "\n".join(f"{LETTERS[i]}. {ch[i]}" for i in range(4))
        q += "\nAnswer with the letter of the correct option."
        msgs = [{"role": "user", "content": q}]
        try:
            txt = tok.apply_chat_template(msgs, tokenize=False,
                                          add_generation_prompt=True,
                                          enable_thinking=False)
        except TypeError:
            txt = tok.apply_chat_template(msgs, tokenize=False,
                                          add_generation_prompt=True)
        prompts.append(txt)
        golds.append(int(r["answer"]))

    correct = 0
    samples = []
    for i in range(0, len(prompts), args.batch_size):
        batch = prompts[i:i + args.batch_size]
        enc = tok(batch, return_tensors="pt", padding=True,
                  truncation=True, max_length=2048).to(args.device)
        logits = model(**enc).logits[:, -1, :]  # 最后位置 -> 下一 token
        logp = torch.log_softmax(logits.float(), dim=-1)
        for b in range(len(batch)):
            scores = [max(logp[b, tid].item() for tid in lids[L]) for L in LETTERS]
            pred = int(max(range(4), key=lambda k: scores[k]))
            g = golds[i + b]
            correct += (pred == g)
            if i + b < 5:
                samples.append({"pred": LETTERS[pred], "gold": LETTERS[g],
                                "ok": pred == g})
    acc = correct / len(prompts)
    res = {"task": "general_mmlu", "n": len(prompts), "correct": correct,
           "accuracy": round(acc, 4), "sec": round(time.time() - t0, 1),
           "samples": samples}
    json.dump(res, open(args.out, "w"), ensure_ascii=False, indent=2)
    print(f"[general] accuracy={acc:.4f} ({correct}/{len(prompts)}) "
          f"用时 {res['sec']}s -> {args.out}")


if __name__ == "__main__":
    main()
