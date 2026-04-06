# Training

RF-DETR training is driven by [`configs/training/dataset.yaml`](../configs/training/dataset.yaml): preprocessing (704×704 stretch via `square_resize_div_64` + `resolution`), Roboflow-style augmentations, and `repeat_train: 2` for “two outputs per example”.

**Windows + CUDA**

1. Install PyTorch with CUDA from [pytorch.org](https://pytorch.org/) (match your GPU).
2. Install project train extras:

```bash
uv sync --extra train
```

3. From the repo root:

```bash
uv run aba-train prepare-data
uv run aba-train train --device cuda
```

Use `--dry-run` on `train` to validate `TrainConfig` without downloading weights or training.

**CPU-only**

```bash
uv run aba-train train --device cpu
```

Adjust `batch_size` / `grad_accum_steps` in `dataset.yaml` if you run out of memory.
