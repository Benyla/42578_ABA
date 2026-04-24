"""Saliency map generation for the two-stage RF-DETR cascade pipeline.

Two black-box attribution methods are implemented, both suitable for
transformer-based detectors where gradient-based methods (Grad-CAM, attention
rollout) are not directly applicable due to deformable attention and
NestedTensor internals.

RISE Algorithm (Petsiuk et al., 2018):
    For each detection D with confidence s_D:
    1. Sample N binary masks M_i (upsampled from low-res random grids).
    2. Run the model on image * M_i.
    3. Saliency = (1 / N*p) * sum_i( s_D(image * M_i) * M_i )
    where p is the mask density (fraction of kept pixels).

LIME Algorithm (Ribeiro et al., 2016):
    For each detection D:
    1. Segment the image into superpixels (SLIC or QuickShift).
    2. Sample N binary vectors z_i indicating which segments are kept.
    3. Run the model on each perturbed image (hidden segments → grey fill).
    4. Fit a weighted ridge regression: confidence ~ z_i, weighted by
       proximity of z_i to the all-ones vector (original image).
    5. Regression coefficients = per-segment importance.
    6. Rasterise segment importances back to pixel space.

    LIME is preferred over RISE when:
    - Detections are small (bullets) and RISE mask resolution is insufficient.
    - The image has clear semantic boundaries (rings, contour, hole edges).
    - You need to know *which region* drove the prediction, not just *where*.

This file exposes:
    RISE:
    - SaliencyConfig          – dataclass for RISE hyper-parameters.
    - compute_saliency_rise   – core RISE loop for a single detection.
    - compute_saliency_batch  – convenience wrapper for multiple detections.
    - add_saliency_to_items   – in-place attachment of saliency maps.

    LIME:
    - LimeConfig              – dataclass for LIME hyper-parameters.
    - compute_saliency_lime   – core LIME loop for a single detection.
    - compute_lime_stage1     – stage-aware wrapper for Target detections.
    - compute_lime_stage2     – stage-aware wrapper for bullet/contour detections.

    Shared:
    - overlay_saliency        – blend a heatmap onto a PIL image.
    - saliency_to_heatmap     – convert float32 map to RGBA uint8.

Stage-awareness:
    Stage 1 saliency is computed on the full greyscale image.
    Stage 2 saliency is computed on the padded crop (crop-local coordinates),
    matching exactly the image region the Stage 2 model actually saw.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class SaliencyConfig:
    """Hyper-parameters for the RISE saliency estimator.

    Attributes:
        n_masks:        Number of random masks to sample (higher = smoother).
        mask_res:       Low-resolution grid size used to generate masks.
                        Each mask is a (mask_res x mask_res) Bernoulli sample
                        upsampled to the input image size via bilinear interp.
        mask_density:   Probability p that a cell is kept (not masked out).
                        Typical range: 0.3–0.6.
        batch_size:     How many masked images to forward in a single call.
                        Lower values reduce peak memory; higher values amortise
                        Python/model overhead.
        score_threshold: Minimum confidence for the *reference* detection to
                         be counted as "still detected" in a masked forward.
                         Should match or be slightly below the pipeline threshold.
        iou_threshold:  IoU required between a masked-forward detection and the
                         reference bbox to be considered the same detection.
        normalise:      If True, normalise the final saliency map to [0, 1].
        seed:           Optional RNG seed for reproducibility.
    """

    n_masks: int = 512
    mask_res: int = 8
    mask_density: float = 0.5
    batch_size: int = 32
    score_threshold: float = 0.25
    iou_threshold: float = 0.4
    normalise: bool = True
    seed: int | None = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _generate_masks(
    n: int,
    img_h: int,
    img_w: int,
    mask_res: int,
    density: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return float32 masks of shape (n, img_h, img_w) in [0, 1].

    Each mask is sampled on a (mask_res x mask_res) Bernoulli grid then
    bilinearly upsampled to (img_h, img_w) — this produces smooth spatial
    blobs rather than sharp pixel-level noise.
    """
    low = rng.random((n, mask_res, mask_res)).astype(np.float32)
    binary_low = (low < density).astype(np.float32)

    # Upsample via PIL (bilinear).  We iterate in chunks to avoid large
    # intermediate tensors.
    masks = np.empty((n, img_h, img_w), dtype=np.float32)
    for i in range(n):
        pil_low = Image.fromarray((binary_low[i] * 255).astype(np.uint8), mode="L")
        pil_up = pil_low.resize((img_w, img_h), Image.BILINEAR)
        masks[i] = np.asarray(pil_up, dtype=np.float32) / 255.0
    return masks


def _apply_mask(image_arr: np.ndarray, mask: np.ndarray) -> Image.Image:
    """Multiply HxWx3 uint8 array by a HxW float mask → PIL Image."""
    masked = (image_arr * mask[:, :, np.newaxis]).clip(0, 255).astype(np.uint8)
    return Image.fromarray(masked)


def _iou(a: list[float], b: list[float]) -> float:
    """Compute IoU between two xyxy boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _detection_score_for_box(
    detections: list[Any],
    ref_box: list[float],
    ref_label: str,
    iou_threshold: float,
) -> float:
    """Find the highest-confidence detection that matches ref_box and ref_label.

    Returns 0.0 if no matching detection is found.
    """
    best = 0.0
    for det in detections:
        if det.label != ref_label:
            continue
        iou = _iou(det.box_xyxy, ref_box)
        if iou >= iou_threshold:
            best = max(best, det.score)
    return best


# ---------------------------------------------------------------------------
# Core RISE loop
# ---------------------------------------------------------------------------


def compute_saliency_rise(
    image: Image.Image,
    ref_box: list[float],
    ref_label: str,
    predict_fn: Callable[[Image.Image], list[Any]],
    cfg: SaliencyConfig | None = None,
) -> np.ndarray:
    """Compute a RISE saliency map for a single detected bounding box.

    Args:
        image:       PIL RGB image that was passed to the model.
        ref_box:     Reference bounding box [x1, y1, x2, y2] in image coords.
        ref_label:   Class label string (e.g. "Target", "Bullet_3").
        predict_fn:  Callable that accepts a PIL Image and returns a list of
                     DetectionItem-like objects with `.label`, `.score`, and
                     `.box_xyxy` attributes.
        cfg:         SaliencyConfig (defaults applied if None).

    Returns:
        float32 numpy array of shape (H, W) with saliency values.
        If ``cfg.normalise`` is True, values are in [0, 1].
    """
    if cfg is None:
        cfg = SaliencyConfig()

    rng = np.random.default_rng(cfg.seed)
    img_arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
    img_h, img_w = img_arr.shape[:2]

    saliency = np.zeros((img_h, img_w), dtype=np.float64)
    n_batches = math.ceil(cfg.n_masks / cfg.batch_size)
    masks_generated = 0

    for _ in range(n_batches):
        n_this = min(cfg.batch_size, cfg.n_masks - masks_generated)
        if n_this <= 0:
            break

        masks = _generate_masks(n_this, img_h, img_w, cfg.mask_res, cfg.mask_density, rng)

        for k in range(n_this):
            masked_img = _apply_mask(img_arr, masks[k])
            dets = predict_fn(masked_img)
            s = _detection_score_for_box(dets, ref_box, ref_label, cfg.iou_threshold)
            saliency += s * masks[k].astype(np.float64)

        masks_generated += n_this

    # Normalise by N * p (RISE weighting factor)
    denom = cfg.n_masks * cfg.mask_density
    saliency /= max(denom, 1e-8)

    if cfg.normalise:
        lo, hi = saliency.min(), saliency.max()
        if hi > lo:
            saliency = (saliency - lo) / (hi - lo)
        else:
            saliency = np.zeros_like(saliency)

    return saliency.astype(np.float32)


# ---------------------------------------------------------------------------
# Batch convenience wrapper
# ---------------------------------------------------------------------------


def compute_saliency_batch(
    image: Image.Image,
    detections: list[Any],
    predict_fn: Callable[[Image.Image], list[Any]],
    cfg: SaliencyConfig | None = None,
) -> dict[int, np.ndarray]:
    """Compute RISE saliency for every detection in *detections*.

    Args:
        image:       PIL RGB image used for detection.
        detections:  List of DetectionItem-like objects.
        predict_fn:  Callable (same signature as in compute_saliency_rise).
        cfg:         Shared SaliencyConfig.

    Returns:
        Dict mapping detection index → saliency array (H, W float32).
    """
    results: dict[int, np.ndarray] = {}
    for idx, det in enumerate(detections):
        sal = compute_saliency_rise(
            image=image,
            ref_box=det.box_xyxy,
            ref_label=det.label,
            predict_fn=predict_fn,
            cfg=cfg,
        )
        results[idx] = sal
    return results


# ---------------------------------------------------------------------------
# Stage-aware helpers that integrate with the cascade pipeline
# ---------------------------------------------------------------------------


def compute_stage1_saliency(
    full_image: Image.Image,
    target_detection: Any,
    cfg: SaliencyConfig | None = None,
) -> np.ndarray:
    """RISE saliency for a Stage 1 (Target) detection on the full image.

    The returned map has the same spatial dimensions as *full_image*.

    Args:
        full_image:        Full-resolution PIL RGB image.
        target_detection:  A DetectionItem with label="Target".
        cfg:               SaliencyConfig.

    Returns:
        float32 saliency array of shape (full_image.height, full_image.width).
    """
    from aba_rfdetr.inference import _get_stage1_model, _MODEL_LOCK, _to_greyscale_rgb

    def _predict(img: Image.Image) -> list[Any]:
        grey = _to_greyscale_rgb(img)
        with _MODEL_LOCK:
            model = _get_stage1_model()
        s1_cfg_raw = {}
        try:
            from aba_rfdetr.inference import load_inference_config
            s1_cfg_raw = load_inference_config().get("stage1", {})
        except Exception:
            pass
        threshold = float(s1_cfg_raw.get("score_threshold", 0.35))
        det = model.predict(grey, threshold=threshold)
        if det is None or len(det) == 0:
            return []
        items = []
        for i in range(len(det)):
            x1, y1, x2, y2 = det.xyxy[i].tolist()

            class _Item:
                pass

            item = _Item()
            item.label = "Target"  # type: ignore[attr-defined]
            item.score = float(det.confidence[i])  # type: ignore[attr-defined]
            item.box_xyxy = [x1, y1, x2, y2]  # type: ignore[attr-defined]
            items.append(item)
        return items

    return compute_saliency_rise(
        image=full_image,
        ref_box=target_detection.box_xyxy,
        ref_label="Target",
        predict_fn=_predict,
        cfg=cfg,
    )


def compute_stage2_saliency(
    crop_image: Image.Image,
    bullet_detection: Any,
    cfg: SaliencyConfig | None = None,
) -> np.ndarray:
    """RISE saliency for a Stage 2 (bullet/contour) detection on a crop image.

    *bullet_detection.box_xyxy* must already be in **crop-local** coordinates
    (i.e. relative to *crop_image*, not the full image).

    Args:
        crop_image:        PIL RGB crop image (as passed to Stage 2).
        bullet_detection:  A DetectionItem in crop-local coordinates.
        cfg:               SaliencyConfig.

    Returns:
        float32 saliency array of shape (crop_image.height, crop_image.width).
    """
    from aba_rfdetr.inference import (
        _get_stage2_model, _MODEL_LOCK, _to_greyscale_rgb,
        _stage2_class_names,
    )

    def _predict(img: Image.Image) -> list[Any]:
        grey = _to_greyscale_rgb(img)
        with _MODEL_LOCK:
            model = _get_stage2_model()
        s2_cfg_raw = {}
        try:
            from aba_rfdetr.inference import load_inference_config
            s2_cfg_raw = load_inference_config().get("stage2", {})
        except Exception:
            pass
        threshold = float(s2_cfg_raw.get("score_threshold", 0.35))
        det = model.predict(grey, threshold=threshold)
        if det is None or len(det) == 0:
            return []
        s2_names = _stage2_class_names()
        items = []
        for j in range(len(det)):
            bx1, by1, bx2, by2 = det.xyxy[j].tolist()
            cid = int(det.class_id[j])
            label = s2_names[cid] if 0 <= cid < len(s2_names) else f"class_{cid}"

            class _Item:
                pass

            item = _Item()
            item.label = label  # type: ignore[attr-defined]
            item.score = float(det.confidence[j])  # type: ignore[attr-defined]
            item.box_xyxy = [bx1, by1, bx2, by2]  # type: ignore[attr-defined]
            items.append(item)
        return items

    return compute_saliency_rise(
        image=crop_image,
        ref_box=bullet_detection.box_xyxy,
        ref_label=bullet_detection.label,
        predict_fn=_predict,
        cfg=cfg,
    )


# ---------------------------------------------------------------------------
# Overlay / visualisation utilities
# ---------------------------------------------------------------------------


def saliency_to_heatmap(
    saliency: np.ndarray,
    colormap: str = "jet",
    alpha: float = 0.5,
) -> np.ndarray:
    """Convert a (H, W) float saliency map to a uint8 RGBA heatmap.

    Args:
        saliency:   float32 array in [0, 1].
        colormap:   Matplotlib colormap name (default "jet").
        alpha:      Overall opacity of the heatmap layer (0 = transparent).

    Returns:
        uint8 RGBA array of shape (H, W, 4).
    """
    import matplotlib.cm as cm  # local import — matplotlib is optional

    cmap = cm.get_cmap(colormap)
    rgba = cmap(saliency)  # (H, W, 4) float in [0, 1]
    rgba[..., 3] = alpha * saliency  # modulate alpha by saliency magnitude
    return (rgba * 255).astype(np.uint8)


def overlay_saliency(
    image: Image.Image,
    saliency: np.ndarray,
    colormap: str = "jet",
    alpha: float = 0.55,
) -> Image.Image:
    """Return a PIL Image with the saliency heatmap blended onto *image*.

    Args:
        image:     Base PIL RGB image.
        saliency:  (H, W) float32 saliency map (must match image dimensions).
        colormap:  Matplotlib colormap name.
        alpha:     Blending strength.

    Returns:
        PIL RGBA Image with the overlay applied.
    """
    heatmap = saliency_to_heatmap(saliency, colormap=colormap, alpha=alpha)
    heat_pil = Image.fromarray(heatmap, mode="RGBA")
    base_rgba = image.convert("RGBA")
    return Image.alpha_composite(base_rgba, heat_pil)


# ---------------------------------------------------------------------------
# Schema extension helper
# ---------------------------------------------------------------------------


def add_saliency_to_items(
    items: list[Any],
    full_image: Image.Image,
    crop_map: dict[int, tuple[Image.Image, tuple[int, int, int, int]]] | None = None,
    cfg: SaliencyConfig | None = None,
) -> None:
    """Attach saliency arrays in-place to each DetectionItem.

    After this call, each item will have a ``saliency`` attribute (numpy
    float32 array) and an ``saliency_overlay`` attribute (PIL RGBA Image).

    For Target detections, saliency is computed on *full_image*.
    For bullet/contour detections, saliency is computed on the crop.  You
    must supply *crop_map* — a dict mapping detection index to (crop_PIL,
    crop_box_xyxy) so the function can convert back to crop-local coords.

    Args:
        items:      List of DetectionItem-like objects (mutated in place).
        full_image: Full-resolution PIL RGB image.
        crop_map:   Optional dict: {item_index: (crop_image, (cx1,cy1,cx2,cy2))}.
                    Required for Stage 2 items.
        cfg:        SaliencyConfig.
    """
    for idx, item in enumerate(items):
        if item.label == "Target":
            sal = compute_stage1_saliency(full_image, item, cfg=cfg)
            item.saliency = sal  # type: ignore[attr-defined]
            item.saliency_overlay = overlay_saliency(full_image, sal)  # type: ignore[attr-defined]
        else:
            if crop_map is None or idx not in crop_map:
                item.saliency = None  # type: ignore[attr-defined]
                item.saliency_overlay = None  # type: ignore[attr-defined]
                continue
            crop_img, (cx1, cy1, _, _) = crop_map[idx]
            # Convert full-image coords → crop-local coords
            local_box = [
                item.box_xyxy[0] - cx1,
                item.box_xyxy[1] - cy1,
                item.box_xyxy[2] - cx1,
                item.box_xyxy[3] - cy1,
            ]

            class _LocalDet:
                pass

            local_det = _LocalDet()
            local_det.label = item.label  # type: ignore[attr-defined]
            local_det.score = item.score  # type: ignore[attr-defined]
            local_det.box_xyxy = local_box  # type: ignore[attr-defined]

            sal = compute_stage2_saliency(crop_img, local_det, cfg=cfg)
            item.saliency = sal  # type: ignore[attr-defined]
            item.saliency_overlay = overlay_saliency(crop_img, sal)  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Integration shim: saliency-augmented staged prediction
# ---------------------------------------------------------------------------


def predict_pil_image_staged_with_saliency(
    image: Image.Image,
    saliency_cfg: SaliencyConfig | None = None,
) -> dict[str, Any]:
    """Run the full staged pipeline and attach RISE saliency maps.

    This is a convenience wrapper that calls ``predict_pil_image_staged`` from
    the core inference module and then computes saliency for every detection,
    returning a dict with both the ``StagedPredictResponse`` and an
    ``saliency_results`` list.

    Each entry in ``saliency_results`` has:
        - ``detection_index``  : int
        - ``label``            : str
        - ``stage``            : 1 (Target) or 2 (bullet/contour)
        - ``saliency``         : float32 numpy array
        - ``overlay_base64``   : JPEG base64 string of the overlay image

    Args:
        image:         PIL RGB image.
        saliency_cfg:  SaliencyConfig (uses defaults if None).

    Returns:
        dict with keys "staged_response" and "saliency_results".
    """
    import base64
    import io

    from aba_rfdetr.inference import predict_pil_image_staged

    staged = predict_pil_image_staged(image)
    saliency_results = []

    # --- Stage 1 saliency ---
    for s1_det in staged.stage1_detections:
        sal = compute_stage1_saliency(image, s1_det, cfg=saliency_cfg)
        overlay = overlay_saliency(image, sal)
        buf = io.BytesIO()
        overlay.save(buf, format="JPEG", quality=85)
        saliency_results.append(
            {
                "detection_index": staged.stage1_detections.index(s1_det),
                "label": s1_det.label,
                "stage": 1,
                "saliency": sal,
                "overlay_base64": base64.b64encode(buf.getvalue()).decode("ascii"),
            }
        )

    # --- Stage 2 saliency (crop-local) ---
    for crop_result in staged.crops:
        cx1, cy1 = int(crop_result.crop_box_xyxy[0]), int(crop_result.crop_box_xyxy[1])
        import io as _io
        crop_bytes = base64.b64decode(crop_result.crop_image_base64)
        crop_pil = Image.open(_io.BytesIO(crop_bytes)).convert("RGB")

        for s2_det in crop_result.detections:
            sal = compute_stage2_saliency(crop_pil, s2_det, cfg=saliency_cfg)
            overlay = overlay_saliency(crop_pil, sal)
            buf = io.BytesIO()
            overlay.save(buf, format="JPEG", quality=85)
            saliency_results.append(
                {
                    "detection_index": crop_result.crop_index,
                    "label": s2_det.label,
                    "stage": 2,
                    "crop_index": crop_result.crop_index,
                    "saliency": sal,
                    "overlay_base64": base64.b64encode(buf.getvalue()).decode("ascii"),
                }
            )

    return {"staged_response": staged, "saliency_results": saliency_results}


# ===========================================================================
# LIME
# ===========================================================================

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class LimeConfig:
    """Hyper-parameters for the LIME saliency estimator.

    Attributes:
        n_samples:          Number of perturbed images to sample.  More samples
                            give a more stable linear fit.  Typical: 64–512.
        segmentation:       Superpixel algorithm.  "slic" (default) respects
                            colour/intensity boundaries and works well for
                            shooting-target images.  "quickshift" produces
                            fewer, larger segments — better for very small
                            crops where SLIC over-segments.
        n_segments:         Target number of superpixel segments.  Fewer
                            segments = coarser but more stable attribution.
                            For Stage 2 crops (small) use 20–40; for full
                            images use 80–150.
        compactness:        SLIC compactness parameter.  Higher values enforce
                            more square segments (less boundary-following).
        fill_value:         Pixel value used to replace hidden segments.
                            127 (mid-grey) is typical; 0 (black) can be used
                            to match the greyscale conversion in inference.
        ridge_alpha:        L2 regularisation for the weighted ridge regression.
                            Increase if coefficients are unstable.
        kernel_width:       Controls the locality kernel: samples with more
                            segments hidden are down-weighted exponentially.
                            Smaller = more local explanation.
        iou_threshold:      IoU required to match a detection to the reference
                            box in a perturbed forward pass.
        score_threshold:    Minimum confidence to count as "detected" in a
                            perturbed image.
        normalise:          If True, normalise the final map to [0, 1].
        positive_only:      If True, zero-out negative LIME coefficients before
                            rasterising (shows only supporting evidence).
        seed:               Optional RNG seed for reproducibility.
    """

    n_samples: int = 256
    segmentation: str = "slic"           # "slic" | "quickshift"
    n_segments: int = 80
    compactness: float = 10.0
    fill_value: int = 127
    ridge_alpha: float = 1.0
    kernel_width: float = 0.25
    iou_threshold: float = 0.4
    score_threshold: float = 0.25
    normalise: bool = True
    positive_only: bool = False
    seed: int | None = None


# ---------------------------------------------------------------------------
# Segmentation helpers
# ---------------------------------------------------------------------------


def _segment_image(image: Image.Image, cfg: LimeConfig) -> np.ndarray:
    """Return an integer label array (H, W) of superpixel segments.

    Requires scikit-image.  ImportError is raised with a helpful message if
    it is not installed.
    """
    try:
        from skimage.segmentation import slic, quickshift
    except ImportError as exc:
        raise ImportError(
            "scikit-image is required for LIME segmentation.\n"
            "Install it with:  uv add scikit-image  (or pip install scikit-image)"
        ) from exc

    arr = np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0
    if cfg.segmentation == "quickshift":
        return quickshift(arr, kernel_size=4, max_dist=200, ratio=0.2)
    # default: slic
    return slic(
        arr,
        n_segments=cfg.n_segments,
        compactness=cfg.compactness,
        start_label=0,
        channel_axis=2,
    )


def _hide_segments(
    image_arr: np.ndarray,
    segments: np.ndarray,
    active: np.ndarray,
    fill: int,
) -> Image.Image:
    """Return a PIL image with inactive segments replaced by *fill* grey.

    Args:
        image_arr:  HxWx3 uint8 numpy array.
        segments:   HxW integer segment label array.
        active:     Boolean array of shape (n_segments,); True = keep segment.
        fill:       Grey fill value for hidden segments.
    """
    out = image_arr.copy()
    mask = active[segments]          # (H, W) bool
    out[~mask] = fill
    return Image.fromarray(out)


# ---------------------------------------------------------------------------
# Weighted ridge regression
# ---------------------------------------------------------------------------


def _fit_lime_weights(
    z: np.ndarray,          # (n_samples, n_segments) binary float
    scores: np.ndarray,     # (n_samples,) confidence values
    kernel_width: float,
    ridge_alpha: float,
) -> np.ndarray:
    """Fit weighted ridge regression; return per-segment coefficients.

    The locality kernel down-weights samples that hide many segments
    (i.e. are far from the original image in segment space).

    w_i = exp( -d_i^2 / kernel_width^2 )
    where d_i = cosine distance between z_i and the all-ones vector.
    """
    try:
        from sklearn.linear_model import Ridge
        from sklearn.metrics.pairwise import cosine_distances

        ones = np.ones((1, z.shape[1]))
        distances = cosine_distances(z, ones).ravel()          # (n_samples,)
        weights = np.exp(-(distances ** 2) / (kernel_width ** 2))

        reg = Ridge(alpha=ridge_alpha, fit_intercept=True)
        reg.fit(z, scores, sample_weight=weights)
        return reg.coef_                                        # (n_segments,)

    except ImportError:
        # Pure-numpy fallback: weighted least squares (no sklearn required)
        ones = np.ones((1, z.shape[1]))
        # cosine distance approximation
        z_norm = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-8)
        distances = 1.0 - z_norm.sum(axis=1) / (z.shape[1] ** 0.5 + 1e-8)
        w = np.exp(-(distances ** 2) / (kernel_width ** 2))

        W = np.diag(w)
        Zt = z.T
        # Closed-form ridge: (Z^T W Z + alpha I)^{-1} Z^T W y
        A = Zt @ W @ z + ridge_alpha * np.eye(z.shape[1])
        b = Zt @ W @ scores
        return np.linalg.solve(A, b)


# ---------------------------------------------------------------------------
# Core LIME loop
# ---------------------------------------------------------------------------


def compute_saliency_lime(
    image: Image.Image,
    ref_box: list[float],
    ref_label: str,
    predict_fn: Callable[[Image.Image], list[Any]],
    cfg: LimeConfig | None = None,
) -> np.ndarray:
    """Compute a LIME saliency map for a single detected bounding box.

    Args:
        image:       PIL RGB image that was passed to the model.
        ref_box:     Reference bounding box [x1, y1, x2, y2] in image coords.
        ref_label:   Class label string (e.g. "Target", "Bullet_3").
        predict_fn:  Callable accepting a PIL Image, returning a list of
                     objects with .label, .score, .box_xyxy attributes.
        cfg:         LimeConfig (defaults applied if None).

    Returns:
        float32 numpy array of shape (H, W).
        Positive values indicate segments that *support* the detection;
        negative values indicate segments that *suppress* it.
        If cfg.positive_only is True, negative values are zeroed out.
        If cfg.normalise is True, the map is scaled to [0, 1].
    """
    if cfg is None:
        cfg = LimeConfig()

    rng = np.random.default_rng(cfg.seed)
    img_arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
    img_h, img_w = img_arr.shape[:2]

    # 1. Segment
    segments = _segment_image(image, cfg)
    n_segs = int(segments.max()) + 1

    # 2. Sample binary perturbation vectors
    #    Each row z_i[k] = 1 means segment k is *shown* (not hidden).
    z = rng.integers(0, 2, size=(cfg.n_samples, n_segs)).astype(np.float32)
    # Always include the unperturbed image as the first sample (z = all-ones)
    z[0] = 1.0

    # 3. Forward pass for each perturbation
    scores = np.zeros(cfg.n_samples, dtype=np.float32)
    for i in range(cfg.n_samples):
        active = z[i].astype(bool)
        perturbed = _hide_segments(img_arr, segments, active, cfg.fill_value)
        dets = predict_fn(perturbed)
        scores[i] = _detection_score_for_box(
            dets, ref_box, ref_label, cfg.iou_threshold
        )

    # 4. Weighted ridge regression → per-segment coefficients
    coefs = _fit_lime_weights(z, scores, cfg.kernel_width, cfg.ridge_alpha)

    # 5. Rasterise coefficients back to pixel space
    if cfg.positive_only:
        coefs = np.clip(coefs, 0, None)

    pixel_map = coefs[segments].astype(np.float32)   # (H, W)

    if cfg.normalise:
        lo, hi = pixel_map.min(), pixel_map.max()
        if hi > lo:
            pixel_map = (pixel_map - lo) / (hi - lo)
        else:
            pixel_map = np.zeros_like(pixel_map)

    return pixel_map


# ---------------------------------------------------------------------------
# Stage-aware LIME wrappers  (mirror of the RISE equivalents above)
# ---------------------------------------------------------------------------


def compute_lime_stage1(
    full_image: Image.Image,
    target_detection: Any,
    cfg: LimeConfig | None = None,
) -> np.ndarray:
    """LIME attribution for a Stage 1 (Target) detection on the full image.

    Returns a float32 array of shape (full_image.height, full_image.width).
    """
    from aba_rfdetr.inference import _get_stage1_model, _MODEL_LOCK, _to_greyscale_rgb

    def _predict(img: Image.Image) -> list[Any]:
        grey = _to_greyscale_rgb(img)
        with _MODEL_LOCK:
            model = _get_stage1_model()
        try:
            from aba_rfdetr.inference import load_inference_config
            s1_cfg_raw = load_inference_config().get("stage1", {})
        except Exception:
            s1_cfg_raw = {}
        threshold = float(s1_cfg_raw.get("score_threshold", 0.35))
        det = model.predict(grey, threshold=threshold)
        if det is None or len(det) == 0:
            return []
        items = []
        for i in range(len(det)):
            x1, y1, x2, y2 = det.xyxy[i].tolist()

            class _Item:
                pass

            item = _Item()
            item.label = "Target"
            item.score = float(det.confidence[i])
            item.box_xyxy = [x1, y1, x2, y2]
            items.append(item)
        return items

    return compute_saliency_lime(
        image=full_image,
        ref_box=target_detection.box_xyxy,
        ref_label="Target",
        predict_fn=_predict,
        cfg=cfg,
    )


def compute_lime_stage2(
    crop_image: Image.Image,
    bullet_detection: Any,
    cfg: LimeConfig | None = None,
) -> np.ndarray:
    """LIME attribution for a Stage 2 (bullet/contour) detection on a crop.

    *bullet_detection.box_xyxy* must be in crop-local coordinates.
    Returns a float32 array of shape (crop_image.height, crop_image.width).
    """
    from aba_rfdetr.inference import (
        _get_stage2_model, _MODEL_LOCK, _to_greyscale_rgb,
        _stage2_class_names,
    )

    def _predict(img: Image.Image) -> list[Any]:
        grey = _to_greyscale_rgb(img)
        with _MODEL_LOCK:
            model = _get_stage2_model()
        try:
            from aba_rfdetr.inference import load_inference_config
            s2_cfg_raw = load_inference_config().get("stage2", {})
        except Exception:
            s2_cfg_raw = {}
        threshold = float(s2_cfg_raw.get("score_threshold", 0.35))
        det = model.predict(grey, threshold=threshold)
        if det is None or len(det) == 0:
            return []
        s2_names = _stage2_class_names()
        items = []
        for j in range(len(det)):
            bx1, by1, bx2, by2 = det.xyxy[j].tolist()
            cid = int(det.class_id[j])
            label = s2_names[cid] if 0 <= cid < len(s2_names) else f"class_{cid}"

            class _Item:
                pass

            item = _Item()
            item.label = label
            item.score = float(det.confidence[j])
            item.box_xyxy = [bx1, by1, bx2, by2]
            items.append(item)
        return items

    return compute_saliency_lime(
        image=crop_image,
        ref_box=bullet_detection.box_xyxy,
        ref_label=bullet_detection.label,
        predict_fn=_predict,
        cfg=cfg,
    )
