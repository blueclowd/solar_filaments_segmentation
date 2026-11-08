#!/usr/bin/env bash
# Final two-stage pipeline on all images: YOLO detector, per-filament U-Net, test submission.
# Run in the background so it survives closing the browser:
#
#   export FILSEG_DATA_ROOT=/mnt/workspace/data/filament-segmentation-2026
#   nohup bash scripts/run_final.sh > final.log 2>&1 &
#   tail -f final.log
#
# Extra flags for the per-filament U-Net go in SEG_EXTRA, e.g. SEG_EXTRA="--lovasz-weight 1.0".
set -euo pipefail

: "${FILSEG_DATA_ROOT:?export FILSEG_DATA_ROOT=<folder holding the downloaded competition data>}"
RUNS=${RUNS:-/mnt/workspace/runs}
NGPU=$(python -c "import torch; print(max(torch.cuda.device_count(), 1))")
DEVICES=$(seq -s, 0 $((NGPU - 1)))
YOLO_BATCH=${YOLO_BATCH:-$((4 * NGPU))}
YOLO_EPOCHS=150
echo "GPUs: $NGPU (device=$DEVICES)  YOLO batch: $YOLO_BATCH  runs: $RUNS"

# last.pt is written every epoch, so only a results.csv with all epochs means YOLO finished
RESULTS="$RUNS/yolo_full/train/results.csv"
DONE_EPOCHS=$( [ -f "$RESULTS" ] && echo $(( $(wc -l < "$RESULTS") - 1 )) || echo 0 )
if [ "$DONE_EPOCHS" -lt "$YOLO_EPOCHS" ] || [ -n "${RETRAIN_YOLO:-}" ]; then
  echo "training YOLO (found $DONE_EPOCHS/$YOLO_EPOCHS finished epochs; starting from scratch)"
  rm -rf "$RUNS/yolo_full/train"
  python train_yolo.py --full --model yolo11m.pt --imgsz 1536 --epochs "$YOLO_EPOCHS" --batch "$YOLO_BATCH" \
    --out-dir "$RUNS/yolo_full" scale=0.25 "device=$DEVICES"
fi

python train_instance.py --full --config configs/instance_r34.yaml --output-dir "$RUNS/instance_full" \
  --encoder resnet34 --epochs 20 --batch-size 32 --crop-size 256 ${SEG_EXTRA:-}

python predict_instances.py --seg-run "$RUNS/instance_full" --yolo-weights "$RUNS/yolo_full/train/weights/last.pt" \
  --split test --settings configs/final_settings.json --out "$RUNS/submission_final.csv"
echo "submission: $RUNS/submission_final.csv"
