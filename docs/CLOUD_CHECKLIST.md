# Google Cloud checklist

Use this when you deploy the API to GCP. Region names below are examples—substitute your own.

## Prerequisites

- [ ] Install [Google Cloud SDK](https://cloud.google.com/sdk/docs/install) (`gcloud`).
- [ ] Create or select a GCP **project**; enable **billing**.
- [ ] Pick a **region** (e.g. `europe-west1`) and use it consistently below.

## Enable APIs

In [APIs & Services](https://console.cloud.google.com/apis/library) enable at least:

- [ ] **Artifact Registry API**
- [ ] **Cloud Run Admin API**
- [ ] **Cloud Build API** (optional, if you use `cloudbuild.yaml`)

## Artifact Registry (Docker images)

- [ ] Create a Docker repository, e.g. `aba-rfdetr`, in your region.
- [ ] Authenticate Docker to Artifact Registry:

```bash
gcloud auth configure-docker REGION-docker.pkg.dev
```

Replace `REGION` (e.g. `europe-west1`).

## Build and push the image

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

Alternatively use [`cloudbuild.yaml`](../cloudbuild.yaml): adjust `_REGION`, `_REPO`, `_IMAGE`, then `gcloud builds submit --config cloudbuild.yaml .`

## Deploy to Cloud Run

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

## Environment variables (inference)

Set as needed on the Cloud Run service:

| Variable | Purpose |
|----------|---------|
| `MODEL_PATH` | Optional override for checkpoint file path inside the container (bake weights into the image or mount later). |
| `ABA_CONFIG_PATH` | Optional absolute path to a custom `inference.yaml` if not using the default under `/app/configs`. |
| `MPLCONFIGDIR` | Already set to `/tmp/matplotlib` in Docker to avoid read-only home issues. |

Increase **`--memory`** (e.g. 4Gi–8Gi) if the process is OOM-killed during PyTorch load or inference.

## Optional: GPU on Cloud Run

RF-DETR can be slow on CPU. Cloud Run supports GPUs in **limited regions**; see [Cloud Run GPU](https://cloud.google.com/run/docs/configuring/services/gpu). GPU adds cost and quota requirements—start with CPU for a prototype.

## Optional: secrets

For API keys or private registry credentials later, use [Secret Manager](https://cloud.google.com/secret-manager/docs) and mount secrets as environment variables or files on Cloud Run.

## CORS

The browser UI is served from the **same** Cloud Run URL as `/predict`, so you typically **do not** need CORS. If you split frontend and API later, add CORS middleware to FastAPI.

## Vertex AI (training)

Training in GCP (Vertex AI custom jobs, managed datasets) is **out of scope** for this scaffold; only the local `configs/training/` and `data/` layout is provided. You can add Vertex pipelines later using the same container or a separate training image.
