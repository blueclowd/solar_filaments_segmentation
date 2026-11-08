# Solar filament instance segmentation (Solar Filament Segmentation Challenge 2026)

Two-stage instance segmentation of solar filaments in 2048×2048 full-disk H-alpha images (MAGFiLO):

1. **Detection:** a YOLO11m detector (1536 px input) proposes one box per filament.
2. **Segmentation:** a U-Net with a ResNet-34 encoder segments each filament from a full-resolution
   256×256 crop around its box. The input has two channels, the image crop and a box mask, so the network
   knows which filament to segment when several are in view.
3. **Assembly:** masks are pasted back from most to least confident box; duplicate masks are removed
   by mask NMS and contested pixels go to the higher probability.

Technical report (LaTeX, competition template): [`report/latex/main.tex`](report/latex/main.tex).

## Results

| Method | Validation PQ | Public leaderboard |
|---|---|---|
| Global U-Net + connected components | 0.403 | 0.34 |
| Two-stage, YOLO11m @ 1536 | 0.448 | 0.39 (YOLO11s @ 1024) |
| Two-stage with ground-truth boxes (upper bound) | 0.686 | — |

Validation: 113 images from held-out months, every annotator's labels scored separately. Inference takes
0.31 s per image on one A10 GPU, including flip test-time augmentation.

## Reproduce the submission

Open [`notebooks/reproduce.ipynb`](notebooks/reproduce.ipynb) and run all cells. On Kaggle, attach the
competition data and a GPU. It loads the final weights from `weights/` and writes `submission.csv`
in about a minute, without training. Set `RETRAIN = True` to retrain both models first (about 3.5 h on
one A10).

From a shell:

```bash
pip install -r requirements.txt
export FILSEG_DATA_ROOT=/path/to/competition/data   # folder containing train/ and test/
python predict_instances.py --split test --seg-run weights --seg-checkpoint weights/instance_r34.pt \
    --yolo-weights weights/yolo11m_1536.pt --settings configs/final_settings.json --out submission.csv
```

## Weights

| File | Model | Trained on |
|---|---|---|
| `weights/yolo11m_1536.pt` | YOLO11m detector, single class, 1536 px | all 707 images, one label file per annotator, 150 epochs |
| `weights/instance_r34.pt` | U-Net / ResNet-34, 2 input channels, 256×256 crops | all 707 images, 20 epochs |

Post-processing settings, tuned on the validation split: `configs/final_settings.json`
(detector confidence 0.3, mask threshold 0.6, mask NMS IoU 0.3, minimum area 200 px).

## Training

Model selection and tuning use a validation split grouped by month (filaments persist for days, so
neighbouring images are near-duplicates):

```bash
python train.py --config configs/unet_r34.yaml --output-dir runs/unet_r34          # global U-Net baseline (defines the split)
python train_yolo.py --unet-run runs/unet_r34 --model yolo11m.pt --imgsz 1536 --epochs 150 \
    --out-dir runs/yolo scale=0.25
python train_instance.py --split-from runs/unet_r34 --output-dir runs/instance_r34
python predict_instances.py --seg-run runs/instance_r34 --yolo-weights runs/yolo/train/weights/best.pt \
    --split val --imgsz 1536                                                        # oracle PQ + settings search
```

The final models are retrained on all images with the same settings:

```bash
export FILSEG_DATA_ROOT=/path/to/competition/data
bash scripts/run_final.sh          # YOLO -> per-filament U-Net -> runs/submission_final.csv
```

`scripts/dsw_setup.sh` installs the requirements and downloads the data with the Kaggle API on a fresh
Linux GPU machine. `notebooks/kaggle_train.ipynb` runs the experiments on Kaggle.

## Repository layout

```
filseg/                 library code
  data.py               annotation loading, month-grouped split, rasterisation, datasets
  augment.py            crop-first augmentation (lookup-table intensity, single affine warp)
  instance.py           per-filament windows, box channel, dataset, instance assembly
  model.py, losses.py   U-Net builder; BCE + Dice (optional clDice / Lovász terms)
  metrics.py            panoptic quality (greedy matching at IoU > 0.5), split/merge counts
  inference.py, postprocess.py, hybrid.py, yolo_data.py
train.py                global U-Net (baseline)
train_yolo.py           YOLO detector
train_instance.py       per-filament U-Net
predict_instances.py    two-stage validation tuning and test prediction
predict.py, predict_hybrid.py           earlier pipelines
visualize_*.py          annotation, prediction and annotator-disagreement figures
configs/                YAML configs and final post-processing settings
scripts/                setup and final-training scripts
weights/                final model weights
notebooks/              reproduce.ipynb (submission), kaggle_train.ipynb (experiments)
report/                 technical report and figures
```

## Notes

- Images are read as single-channel grayscale. Filament classes (chirality) are merged into one class,
  and only the segmentation masks are used for training.
- 296 training images have two or more independent annotations. Each annotation set is used as a
  separate training sample and scored separately on validation. Annotators agree on only 58% of
  filaments at IoU > 0.5 (`visualize_disagreement.py`), which limits the achievable score against any
  single annotator.
- Ultralytics YOLO is licensed under AGPL-3.0.
