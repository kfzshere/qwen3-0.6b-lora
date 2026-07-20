#!/bin/bash
# 三方向基线评测并行启动：math(卡0) / safety(卡1) / general(卡2)
set -u
BASE=${BASE:-/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-hl02/hadoop-aipnlp/3A/multimodal/zhufengkai/qwen-demo}
MODEL=${MODEL:-$BASE/models/Qwen3-0.6B}
DS=$BASE/datasets
OUT=$BASE/eval_results
mkdir -p "$OUT"
cd "$(dirname "$0")"   # eval/ 目录，便于 import common

echo "=== 基线评测启动 $(date) ==="
echo "model=$MODEL"

CUDA_VISIBLE_DEVICES=0 python3 eval_math.py \
  --model "$MODEL" --data "$DS/math/gsm8k_test.parquet" \
  --out "$OUT/math.json" --device cuda:0 --thinking \
  > "$OUT/math.log" 2>&1 &
PID_M=$!

CUDA_VISIBLE_DEVICES=1 python3 eval_safety.py \
  --model "$MODEL" --data "$DS/safety/aegis_refusals_val.json" \
  --out "$OUT/safety.json" --device cuda:0 \
  > "$OUT/safety.log" 2>&1 &
PID_S=$!

CUDA_VISIBLE_DEVICES=2 python3 eval_general.py \
  --model "$MODEL" --data "$DS/general_eval/mmlu_test_1000.parquet" \
  --out "$OUT/general.json" --device cuda:0 \
  > "$OUT/general.log" 2>&1 &
PID_G=$!

echo "PIDs: math=$PID_M safety=$PID_S general=$PID_G"
wait $PID_M $PID_S $PID_G
echo "=== 全部完成 $(date) ==="
for f in math safety general; do
  echo "--- $f ---"; cat "$OUT/$f.json" 2>/dev/null | python3 -c "import sys,json;d=json.load(sys.stdin);print({k:v for k,v in d.items() if k!='samples'})" 2>/dev/null || echo "(无结果，看 $OUT/$f.log)"
done
