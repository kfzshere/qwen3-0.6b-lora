"""手写 TIES-Merging 合并多个专家模型（safetensors 权重级）。
TIES: 对每个专家的 task vector(专家权重-基座) 做 trim(保留每张量Top-k%绝对值),
选举主导符号, 只平均与主导符号一致的更新, 加回基座。
用法: python ties_merge.py --base BASE --experts m1:1.2 m2:1.0 m3:1.0 --density 0.6 --out OUT
"""
import argparse, os, json, glob
import torch
from safetensors.torch import load_file, save_file


def load_sd(path):
    sd = {}
    for f in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
        sd.update(load_file(f))
    return sd


def trim(delta, density):
    """每张量保留绝对值 Top-(density) 的元素, 其余置零。"""
    if density >= 1.0:
        return delta
    flat = delta.abs().flatten()
    k = max(1, int(flat.numel() * density))
    thresh = torch.topk(flat, k, largest=True).values.min()
    return torch.where(delta.abs() >= thresh, delta, torch.zeros_like(delta))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--experts", nargs="+", required=True, help="path:weight ...")
    ap.add_argument("--density", type=float, default=0.6)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    specs = []
    for e in args.experts:
        p, w = e.rsplit(":", 1)
        specs.append((p, float(w)))

    print(f"[ties] base={args.base}, {len(specs)} experts, density={args.density}")
    base = load_sd(args.base)
    # 预加载专家
    experts = [(load_sd(p), w) for p, w in specs]

    merged = {}
    for key in base:
        b = base[key]
        if b.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            merged[key] = b
            continue
        bf = b.float()
        # 各专家 task vector, trim
        tvs, ws = [], []
        for sd, w in experts:
            if key not in sd:
                continue
            delta = sd[key].float() - bf
            tvs.append(trim(delta, args.density) * w)
            ws.append(w)
        if not tvs:
            merged[key] = b
            continue
        stack = torch.stack(tvs)  # [n, ...]
        # 主导符号：加权和的符号
        sign = torch.sign(stack.sum(dim=0))
        sign[sign == 0] = 1
        # 只保留与主导符号一致的更新
        keep = (torch.sign(stack) == sign.unsqueeze(0)).float()
        agreed = stack * keep
        cnt = keep.sum(dim=0).clamp(min=1)
        disjoint_mean = agreed.sum(dim=0) / cnt
        merged[key] = (bf + disjoint_mean).to(b.dtype)

    os.makedirs(args.out, exist_ok=True)
    save_file(merged, os.path.join(args.out, "model.safetensors"),
              metadata={"format": "pt"})
    # 拷贝配置/tokenizer
    for f in os.listdir(args.base):
        if f.endswith((".json", ".txt")) and f != "model.safetensors.index.json":
            src = os.path.join(args.base, f)
            if os.path.isfile(src):
                os.system(f"cp '{src}' '{args.out}/'")
    print(f"[ties] 合并完成 -> {args.out}")


if __name__ == "__main__":
    main()
