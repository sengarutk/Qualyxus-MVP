#!/bin/bash
# Full VisA benchmark, all 12 categories, one process per category so a crash loses at most one.
# Resumes: finished (category, candidate) pairs in the JSON are skipped.
cd /home/sengar/dev
OUT=reports/visa_full/anomaly_benchmark_visa_all.json
CANDS=dinomaly_dinov2reg_b,patchcore_dinov2_s,patchcore_wrn50
for pass in 1 2; do
for c in candle cashew chewinggum macaroni1 pcb2 pcb3 pcb4 pipe_fryum pcb1 capsules macaroni2 fryum; do
  echo "=== pass $pass $c $(date -u +%FT%TZ)"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -m src.inference.anomaly.benchmark \
    --visa /home/sengar/datasets/visa/VisA --categories $c --candidates $CANDS --select --out $OUT \
    || echo "!!! $c failed with exit $?"
done
done
echo "=== ALL DONE $(date -u +%FT%TZ)"
