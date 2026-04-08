---
title: ABA Shooting Target Detector
emoji: 🎯
colorFrom: gray
colorTo: red
sdk: docker
app_port: 7860
pinned: false
---

# ABA Shooting Target Detector

Two-stage RF-DETR object detection pipeline for automatic classification of shooting target types.

**Stage 1** — Detect the Target region on the full image.
**Stage 2** — Crop to the Target, then detect individual bullet holes and contours.

Upload an image at the web UI or `POST /predict/staged` with a `file` form field.
