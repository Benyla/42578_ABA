# Automatic classification of shooting target types (Advanced Business Analytics)

**Advanced Business Analytics** — *27 March 2026*

**Andreas** (s210986), **Morten** (s214927), **Bertram** (s224194), **Jakob** (s214411)

---

## What this project is about

In military, police, and security training, many different target types are used depending on distance, drill type, and qualification requirements. Misidentifying or misusing targets can create inconsistencies in training, increase instructor workload, and reduce standardization across exercises.

This project proposes a **computer vision system** that recognizes target types from **range images** across **multiple classes**. The purpose is to support **automatic verification** and **digital registration** of targets, so training administration becomes safer, more efficient, and more consistent.

## Research question

**Can a computer vision model reliably classify shooting target types from range images across many classes, and how can such a system support more standardized and resilient training administration?**

We examine more specifically:

- How accurately target types can be classified from real-world images.
- Whether the model can provide useful **decision support** for target verification in practice.
- Which target types or conditions produce the most errors.
- Which methods fit the problem and what **requirements and environment** they need to run in.

## Data and method

The work builds on a dataset of roughly **2,500 images** of shooting targets from training environments, with labels for model training. Because the number of classes is relatively high compared to dataset size, **transfer learning** and **data augmentation** are central.

The analysis includes:

- Deep learning for visual recognition on range imagery.
- **Transfer learning** to adapt a pretrained model to this domain.
- **Uncertainty and error analysis** to find weak points and failure cases.
- **Explainability** to see which visual features drive predictions.

## Expected contribution

The project contributes **technically** (a model for automatic target recognition) and **analytically** (evaluation of where the system works or fails, and how it can improve **resilience** of training administration by reducing manual errors, increasing consistency, and supporting more reliable documentation of training setups).

## Relevance to the course

The project combines **business analytics and resilience** with applied machine learning. It is not only an engineering exercise: we also evaluate **limitations**, **operational usefulness**, and implications for **safer, more standardized** training processes. Themes include computer vision, deep learning, and explainable AI.

---

## This repository

This codebase is the **implementation scaffold** for that work: **RF-DETR**-based detection, a **FastAPI** service with a small upload UI and JSON API, a **COCO-style** data layout (`data/images/` + annotations), and **training** driven by YAML config. It supports the course goal of rigorous evaluation on real range imagery, not only model training.

| Topic | Where to look |
|--------|----------------|
| Run API locally, inference config | [`configs/inference.yaml`](configs/inference.yaml) |
| Training (prepare data, train, augmentations) | [`training/README.md`](training/README.md), [`configs/training/dataset.yaml`](configs/training/dataset.yaml) |
| Data layout | [`data/README.md`](data/README.md) |
| Deploy to Google Cloud | [`docs/CLOUD_CHECKLIST.md`](docs/CLOUD_CHECKLIST.md) |

**Quick start (local API):** requires Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --all-extras
uv run uvicorn aba_rfdetr.api:app --host 0.0.0.0 --port 8080
```

Open `http://127.0.0.1:8080/` to try uploads, or `POST /predict` with form field `file`.

Run tests: `uv run pytest` (optional: `uv run ruff check src tests`).

---

## License

Add a `LICENSE` file if you distribute the project.
