#!/usr/bin/env bash
# One-time setup on Alibaba Cloud PAI-DSW (or any Linux GPU machine): install packages and
# download the competition data with the Kaggle API.
#
#   export KAGGLE_USERNAME=... KAGGLE_KEY=...   # from kaggle.com -> Settings -> API -> Create New Token
#   bash scripts/dsw_setup.sh
set -euo pipefail

DATA_DIR=${DATA_DIR:-/mnt/workspace/data/filament-segmentation-2026}
COMPETITION=filament-segmentation-2026

echo "[1/3] installing packages"
pip install --progress-bar on -r requirements.txt kaggle

echo "[2/3] downloading data to $DATA_DIR"
if [ -z "${KAGGLE_USERNAME:-}" ] && [ ! -f ~/.kaggle/kaggle.json ]; then
  echo "Kaggle credentials missing: export KAGGLE_USERNAME and KAGGLE_KEY, or put kaggle.json in ~/.kaggle/" >&2
  exit 1
fi

if find "$DATA_DIR" -name 'MAGFiLO_1.0_Annotations_kaggle2026_train.json' 2>/dev/null | grep -q .; then
  echo "data already present in $DATA_DIR"
else
  mkdir -p "$DATA_DIR"
  kaggle competitions download -c "$COMPETITION" -p "$DATA_DIR"
  echo "unzipping..."
  unzip -q -o "$DATA_DIR/$COMPETITION.zip" -d "$DATA_DIR"
  rm "$DATA_DIR/$COMPETITION.zip"
fi

echo "[3/3] checking GPUs"
python -c "import torch; print('torch', torch.__version__, '| GPUs:', torch.cuda.device_count())"
echo "next: export FILSEG_DATA_ROOT=$DATA_DIR && bash scripts/run_final.sh"
