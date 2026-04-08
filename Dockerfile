FROM python:3.12-slim-bookworm

RUN apt-get update && \
    apt-get install --no-install-recommends -y \
        build-essential \
        curl \
        libglib2.0-0 \
        libgl1 \
        libsm6 \
        libxext6 \
        libxrender1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 appuser

WORKDIR /app

# CPU-only PyTorch (HF Spaces free tier has no GPU)
RUN pip install --no-cache-dir \
        torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md ./
COPY src src/
COPY configs configs/

# Editable install so __file__ stays under /app/src/ and _PROJECT_ROOT resolves to /app
RUN pip install --no-cache-dir -e .

COPY models/stage1_target.pth models/stage1_target.pth
COPY models/stage2_bullet.pth models/stage2_bullet.pth

RUN chown -R appuser:appuser /app
USER appuser

ENV PYTHONUNBUFFERED=1
ENV MPLCONFIGDIR=/tmp/matplotlib
EXPOSE 7860

CMD ["uvicorn", "aba_rfdetr.api:app", "--host", "0.0.0.0", "--port", "7860"]
