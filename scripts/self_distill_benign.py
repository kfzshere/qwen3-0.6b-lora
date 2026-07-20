"""自蒸馏有用性：用基座模型对"看着敏感但无害"的 prompt 生成直接帮忙的答案，
只保留基座没拒答的（基座过度拒答仅 4.4%），作为良性对照数据治过度拒答。
用法: python self_distill_benign.py --model <基座> --prompts <falsereject_benign.json> --out <benign_sd.json>
"""
import argparse, json, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "eval"))
from common import load_model, chat_generate, is_refusal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=96)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    args = ap.parse_args()

    data = json.load(open(args.prompts))
    if args.limit:
        data = data[:args.limit]
    prompts = [d["instruction"] for d in data]

    print(f"[sd] 基座生成 {len(prompts)} 条良性回答(非thinking)...")
    model, tok = load_model(args.model, args.device)
    outs = chat_generate(model, tok, prompts, args.device,
                         max_new_tokens=args.max_new_tokens,
                         batch_size=args.batch_size, enable_thinking=False)
    kept = []
    for i, out in enumerate(outs):
        o = out.strip()
        # 只保留基座"直接帮忙"（没拒答）且非空的
        if o and not is_refusal(o) and len(o) > 20:
            kept.append({"instruction": prompts[i], "input": "",
                         "output": o, "task": "safety_benign"})
    json.dump(kept, open(args.out, "w"), ensure_ascii=False, indent=1)
    print(f"[sd] 保留直接帮忙 {len(kept)}/{len(prompts)} "
          f"({len(kept)/len(prompts)*100:.1f}%) -> {args.out}")


if __name__ == "__main__":
    main()
