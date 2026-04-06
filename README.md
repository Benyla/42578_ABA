# ABA RF-DETR

Scaffold for **RF-DETR** object detection: a **FastAPI** service that serves a small **upload UI** and **`/predict`** JSON API, a simple **`data/`** layout (images + COCO JSON), and **RF-DETR training** driven by [`configs/training/dataset.yaml`](configs/training/dataset.yaml). Model weights live under `models/` (not committed).

## Quick start (local)

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --all-extras
uv run uvicorn aba_rfdetr.api:app --host 0.0.0.0 --port 8080
```

Open `http://127.0.0.1:8080/` to upload an image, or call `POST /predict` with form field `file` (image).

**Default inference** uses RF-DETR’s built-in pretrained weights (downloaded on first run). To use your own checkpoint, set `checkpoint_path` in [`configs/inference.yaml`](configs/inference.yaml) or set `MODEL_PATH` to a `.pth` file, and set `num_classes` / `model_class` to match training. For custom class order, set `class_names_file` to a JSON list of names (see config comments).

**Training** (PyTorch Lightning + CUDA on Windows; see [`training/README.md`](training/README.md)):

```bash
uv sync --extra train
uv run aba-train prepare-data
uv run aba-train train --device cuda
```

Validate config without training: `uv run aba-train train --dry-run`.

## Project layout

- `src/aba_rfdetr/` — API, inference wrapper, COCO class names, HTML template
- `configs/inference.yaml` — model class, checkpoint path, score threshold
- `configs/training/dataset.yaml` — RF-DETR training (augmentations, `RFDETRMedium` @ 704px, Roboflow layout)
- `data/README.md` — `data/images/` + `data/instances.json`
- `models/` — place `*.pth` weights here (gitignored)
- `training/README.md` — notes for future training scripts
- `dockerfiles/Dockerfile.cloudrun` — image for Google Cloud Run

## Tests

```bash
uv run pytest
uv run ruff check src tests
```

---

## Google Cloud checklist

Complete these steps once per project (region names are examples—use your own).

### Prerequisites

- [ ] Install [Google Cloud SDK](https://cloud.google.com/sdk/docs/install) (`gcloud`).
- [ ] Create or select a GCP **project**; enable **billing**.
- [ ] Pick a **region** (e.g. `europe-west1`) and use it consistently below.

### Enable APIs

In [APIs & Services](https://console.cloud.google.com/apis/library) enable at least:

- [ ] **Artifact Registry API**
- [ ] **Cloud Run Admin API**
- [ ] **Cloud Build API** (optional, if you use `cloudbuild.yaml`)

### Artifact Registry (Docker images)

- [ ] Create a Docker repository, e.g. `aba-rfdetr`, in your region.
- [ ] Authenticate Docker to Artifact Registry:

```bash
gcloud auth configure-docker REGION-docker.pkg.dev
```

Replace `REGION` (e.g. `europe-west1`).

### Build and push the image

From the repository root (with Docker installed):

```bash
export REGION=europe-west1
export PROJECT_ID=$(gcloud config get-value project)
export REPO=aba-rfdetr
export IMAGE=api

docker build -f dockerfiles/Dockerfile.cloudrun \
  -t ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${IMAGE}:v1 .

docker push ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${IMAGE}:v1
```

Alternatively use [`cloudbuild.yaml`](cloudbuild.yaml): adjust `_REGION`, `_REPO`, `_IMAGE`, then `gcloud builds submit --config cloudbuild.yaml .`

### Deploy to Cloud Run

- [ ] Deploy the container (HTTP on **8080** matches the Dockerfile):

```bash
gcloud run deploy aba-rfdetr-api \
  --image ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${IMAGE}:v1 \
  --region ${REGION} \
  --platform managed \
  --port 8080 \
  --memory 2Gi \
  --cpu 2 \
  --timeout 300 \
  --allow-unauthenticated
```

- [ ] If the service must stay **private**, omit `--allow-unauthenticated` and grant `roles/run.invoker` to callers.

### Environment variables (inference)

Set as needed on the Cloud Run service:

| Variable | Purpose |
|----------|---------|
| `MODEL_PATH` | Optional override for checkpoint file path inside the container (bake weights into the image or mount later). |
| `ABA_CONFIG_PATH` | Optional absolute path to a custom `inference.yaml` if not using the default under `/app/configs`. |
| `MPLCONFIGDIR` | Already set to `/tmp/matplotlib` in Docker to avoid read-only home issues. |

Increase **`--memory`** (e.g. 4Gi–8Gi) if the process is OOM-killed during PyTorch load or inference.

### Optional: GPU on Cloud Run

RF-DETR can be slow on CPU. Cloud Run supports GPUs in **limited regions**; see [Cloud Run GPU](https://cloud.google.com/run/docs/configuring/services/gpu). GPU adds cost and quota requirements—start with CPU for a prototype.

### Optional: secrets

For API keys or private registry credentials later, use [Secret Manager](https://cloud.google.com/secret-manager/docs) and mount secrets as environment variables or files on Cloud Run.

### CORS

The browser UI is served from the **same** Cloud Run URL as `/predict`, so you typically **do not** need CORS. If you split frontend and API later, add CORS middleware to FastAPI.

### Vertex AI (training)

Training in GCP (Vertex AI custom jobs, managed datasets) is **out of scope** for this scaffold; only the local `configs/training/` and `data/` layout is provided. You can add Vertex pipelines later using the same container or a separate training image.

---

## License

Add a `LICENSE` file if you distribute the project.
