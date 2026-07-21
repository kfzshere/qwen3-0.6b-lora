"""数学 GRPO (RLVR)：奖励=答案正确性，从 SFT checkpoint 继续用强化学习提升数学。
KL 约束(beta)防止把安全/通用带崩。用法见 argparse。
"""
import argparse, re, os, sys
import pyarrow.parquet as pq
from datasets import Dataset
from trl import GRPOTrainer, GRPOConfig

HINT = "\nPlease reason step by step, and put your final answer after '#### '."


def norm(s):
    if s is None:
        return None
    s = str(s).strip().replace(",", "").replace("$", "").replace("%", "").rstrip(".")
    m = re.search(r"-?\d+\.?\d*", s)
    if not m:
        return None
    try:
        f = float(m.group(0)); return int(f) if f == int(f) else round(f, 4)
    except ValueError:
        return None


def extract_pred(text):
    m = re.search(r"####\s*([^\n]+)", text)
    if m and norm(m.group(1)) is not None:
        return norm(m.group(1))
    b = re.findall(r"\\boxed\{([^}]*)\}", text)
    if b and norm(b[-1]) is not None:
        return norm(b[-1])
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
    return norm(nums[-1]) if nums else None


def get_text(comp):
    if isinstance(comp, list):  # conversational: [{"role":..,"content":..}]
        return comp[-1]["content"]
    return comp


def reward_correct(completions, answer, **kwargs):
    out = []
    for comp, gold in zip(completions, answer):
        pred = extract_pred(get_text(comp))
        out.append(1.0 if (pred is not None and pred == gold) else 0.0)
    return out


def reward_format(completions, **kwargs):
    # 轻微鼓励输出 #### 格式
    return [0.2 if "####" in get_text(c) else 0.0 for c in completions]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--num_gen", type=int, default=8)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--beta", type=float, default=0.04)
    ap.add_argument("--max_steps", type=int, default=150)
    args = ap.parse_args()

    rows = pq.read_table(args.data).to_pylist()[:args.limit]
    def gold(a):
        m = re.search(r"####\s*(.+)", a); return norm(m.group(1)) if m else None
    data = [{"prompt": [{"role": "user", "content": r["question"] + HINT}],
             "answer": gold(r["answer"])} for r in rows]
    data = [d for d in data if d["answer"] is not None]
    ds = Dataset.from_list(data)
    print(f"[grpo] {len(ds)} 题, num_gen={args.num_gen}, lr={args.lr}, beta={args.beta}")

    cfg = GRPOConfig(
        output_dir=args.out,
        per_device_train_batch_size=args.bs,
        num_generations=args.num_gen,
        gradient_accumulation_steps=2,
        learning_rate=args.lr,
        beta=args.beta,
        max_prompt_length=300,
        max_completion_length=512,
        temperature=0.9,
        max_steps=args.max_steps,
        logging_steps=10,
        save_steps=args.max_steps,
        bf16=True,
        gradient_checkpointing=True,
        report_to="none",
        num_iterations=1,
    )
    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=[reward_correct, reward_format],
        args=cfg,
        train_dataset=ds,
    )
    trainer.train()
    trainer.save_model(args.out)
    print(f"[grpo] 完成 -> {args.out}")


if __name__ == "__main__":
    main()
