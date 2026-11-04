#!/usr/bin/env bash
# Extract keypoints -> build tables -> train (all + raw ablation) -> evaluate on held-out clips -> render videos.
# Run from Waving-GD/ :
#   VERSION=v3 FPR=0.06 MIN_RECALL=0.95 SKIP_PREP=1 bash scripts/run_pipeline.sh
# Each trained model is a version folder outputs/runs/<VERSION>_<all|raw>_<tag>/ (train.log, eval_test.txt, videos/).
# MIN_RECALL (optional): choose threshold = highest one with val recall >= MIN_RECALL, else best F1 under the FPR cap.
# SKIP_PREP=1 skips extraction + build_dataset (already done). GPU=0/1 picks the device (default 1).
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${GPU:-1}"
V="${VERSION:?set VERSION, e.g. VERSION=v3}"; FPR="${FPR:-0.05}"; MR="${MIN_RECALL:-0}"
if [ "$MR" != "0" ]; then TAG="rec$(python3 -c "print(round($MR*100))")"; else TAG="fpr$(printf '%02d' "$(python3 -c "print(round($FPR*100))")")"; fi
PY=.venv/bin/python
if [ -z "${SKIP_PREP:-}" ]; then
  $PY -m pipeline.extract_features test
  $PY -m pipeline.extract_features videos
  $PY -m pipeline.extract_features scb
  $PY -m pipeline.build_dataset
fi
$PY -m classifier.train --name ${V}_all_${TAG} --max-hn-fpr $FPR --min-recall $MR
$PY -m classifier.train --name ${V}_raw_${TAG} --max-hn-fpr $FPR --min-recall $MR --features raw
$PY -m classifier.evaluate --name ${V}_all_${TAG} --name ${V}_raw_${TAG}
for m in ${V}_all_${TAG} ${V}_raw_${TAG}; do for c in IMG_8242 IMG_8243; do
  $PY -m pipeline.detector --model $m --source dataset/$c.MOV --window 5 --render; done; done
