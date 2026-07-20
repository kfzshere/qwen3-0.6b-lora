"""安全基线评测：Aegis refusals val 拒答率。"""
import argparse, json, time
from common import load_model, chat_generate, is_refusal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    args = ap.parse_args()

    data = json.load(open(args.data))
    # 只测有害 prompt（prompt_label == unsafe）
    data = [d for d in data if d.get("prompt_label", "unsafe") == "unsafe"]
    if args.limit:
        data = data[:args.limit]
    prompts = [d["prompt"] for d in data]

    print(f"[safety] {len(prompts)} 条有害 prompt, device={args.device}")
    t0 = time.time()
    model, tok = load_model(args.model, args.device)
    outs = chat_generate(model, tok, prompts, args.device,
                         max_new_tokens=args.max_new_tokens,
                         batch_size=args.batch_size, enable_thinking=False)
    refused = 0
    samples = []
    for i, out in enumerate(outs):
        r = is_refusal(out)
        refused += r
        if i < 5:
            samples.append({"prompt": prompts[i][:80], "refused": r, "out": out[:200]})
    rate = refused / len(prompts)
    res = {"task": "safety_aegis", "n": len(prompts), "refused": refused,
           "refusal_rate": round(rate, 4), "sec": round(time.time() - t0, 1),
           "samples": samples}
    json.dump(res, open(args.out, "w"), ensure_ascii=False, indent=2)
    print(f"[safety] refusal_rate={rate:.4f} ({refused}/{len(prompts)}) "
          f"用时 {res['sec']}s -> {args.out}")


if __name__ == "__main__":
    main()
