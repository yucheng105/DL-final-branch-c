# DL-final-branch-c

Branch C: Patch-Level Forensic Branch for AI-generated image detection.

This implementation follows the proposal idea inspired by AIDE: split an image
into local patches, score each patch by frequency energy, select both
high-frequency and low-frequency patches, and learn local forensic features for
real/fake classification.

## Dataset Layout

Expected local layout:

```text
dataset/
  cifake/
    train/
      REAL/
      FAKE/
    test/
      REAL/
      FAKE/
  tiny-genimage/
    imagenet_ai_0419_biggan/
      train/{nature,ai}/
      val/{nature,ai}/
    ...
```

Labels:

- `REAL` / `nature` = `0`
- `FAKE` / `ai` = `1`

## Model

`src/model.py` implements:

```text
image
 -> unfold into patches
 -> FFT high-frequency energy score per patch
 -> top-K high-frequency patches
 -> bottom-K low-frequency patches
 -> high/low CNN patch encoders
 -> attention pooling
 -> fusion classifier
```

For CIFAKE, keep the forensic view at `32x32`:

- `patch_size=16`
- `stride=8`
- `top_k=4`

For tiny-genimage, resize the forensic view and use larger patches:

- `forensic_size=224`
- `patch_size=32`
- `stride=16`
- `top_k=8`

## Quick Smoke Test

```bash
python train.py --dataset cifake --epochs 1 --batch-size 32 --num-workers 0 --max-train-samples 128 --max-val-samples 64 --output-dir runs/smoke
```

## Train on CIFAKE

```bash
python train.py --dataset cifake --epochs 10 --batch-size 128 --patch-size 16 --stride 8 --top-k 4 --output-dir runs/cifake_branch_c
```

Evaluate on CIFAKE test:

```bash
python evaluate.py --checkpoint runs/cifake_branch_c/best.pt --dataset cifake --split test --batch-size 128 --patch-size 16 --stride 8 --top-k 4
```

## Train on Tiny-GenImage

Example: train only on `sdv5`, then test on unseen generators.

```bash
python train.py --dataset tiny-genimage --generators sdv5 --epochs 10 --batch-size 64 --forensic-size 224 --patch-size 32 --stride 16 --top-k 8 --output-dir runs/tiny_sdv5_branch_c
```

Evaluate on unseen generators:

```bash
python evaluate.py --checkpoint runs/tiny_sdv5_branch_c/best.pt --dataset tiny-genimage --split val --generators biggan vqdm wukong adm glide midjourney --batch-size 64 --forensic-size 224 --patch-size 32 --stride 16 --top-k 8
```

## Files

- `src/data.py`: dataset scanning and semantic/forensic transforms
- `src/model.py`: FFT patch selector and patch-level forensic branch
- `src/engine.py`: training, evaluation, checkpoint helpers
- `src/metrics.py`: accuracy, balanced accuracy, precision, recall, F1, AUROC
- `train.py`: training CLI
- `evaluate.py`: evaluation CLI
