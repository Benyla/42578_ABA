"""test_saliency.py — Visual test for the two-stage cascade + RISE saliency.

Usage
-----
    # Quickest: test on a single image, save outputs to ./saliency_output/
    python test_saliency.py --image path/to/shot.jpg

    # Also display plots interactively (requires a display / Jupyter)
    python test_saliency.py --image path/to/shot.jpg --plot

    # Batch: run every image in a folder
    python test_saliency.py --image_dir path/to/images/ --plot

    # Tune RISE for speed vs quality
    python test_saliency.py --image path/to/shot.jpg --n_masks 128 --mask_res 6

    # Only run Stage 1 saliency (skip Stage 2 bullets)
    python test_saliency.py --image path/to/shot.jpg --stage1_only

Output layout (per image)
-------------------------
    saliency_output/
    └── <image_stem>/
        ├── detections.png          full image with bbox overlays
        ├── stage1_target_<N>.png   saliency overlay for each Target
        ├── stage2_crop_<N>.png     padded crop with all Stage 2 bboxes
        ├── stage2_<label>_<N>.png  saliency overlay per bullet detection
        └── summary.png             single figure with all panels side-by-side
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Colour palette for bbox labels
# ---------------------------------------------------------------------------

_LABEL_COLOURS: dict[str, tuple[int, int, int]] = {
    "Target":        (255, 80,  80),
    "Bullet_0":      (80,  200, 120),
    "Bullet_1":      (80,  160, 255),
    "Bullet_2":      (255, 200, 60),
    "Bullet_3":      (200, 80,  255),
    "Bullet_4":      (80,  230, 230),
    "Bullet_5":      (255, 140, 60),
    "Bullet_6":      (160, 255, 80),
    "Bullet_7":      (255, 80,  180),
    "Bullet_8":      (120, 120, 255),
    "Bullet_9":      (255, 220, 120),
    "Bullet_10":     (80,  255, 180),
    "black_contour": (180, 180, 180),
}


def _colour(label: str) -> tuple[int, int, int]:
    return _LABEL_COLOURS.get(label, (220, 220, 220))


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _load_font(size: int = 14):
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _draw_boxes(image: Image.Image, detections: list, title: str = "") -> Image.Image:
    """Return a copy of *image* (RGB) with bboxes and labels drawn on it."""
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    font = _load_font(14)

    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det.box_xyxy]
        colour = _colour(det.label)
        draw.rectangle([x1, y1, x2, y2], outline=colour, width=2)
        tag = f"{det.label} {det.score:.2f}"
        try:
            bbox_t = draw.textbbox((x1, y1 - 16), tag, font=font)
            draw.rectangle(bbox_t, fill=colour)
        except AttributeError:
            pass
        draw.text((x1, y1 - 16), tag, fill=(0, 0, 0), font=font)

    if title:
        draw.text((8, 8), title, fill=(255, 255, 0), font=font)

    return out


def _add_label_bar(image: Image.Image, text: str, height: int = 24) -> Image.Image:
    bar = Image.new("RGB", (image.width, height), (20, 20, 20))
    draw = ImageDraw.Draw(bar)
    font = _load_font(12)
    draw.text((6, 4), text, fill=(220, 220, 220), font=font)
    combined = Image.new("RGB", (image.width, image.height + height))
    combined.paste(image, (0, 0))
    combined.paste(bar, (0, image.height))
    return combined


def _hstack(images: list[Image.Image], gap: int = 4) -> Image.Image:
    if not images:
        return Image.new("RGB", (1, 1))
    target_h = max(im.height for im in images)
    resized = []
    for im in images:
        if im.height != target_h:
            ratio = target_h / im.height
            im = im.resize((int(im.width * ratio), target_h), Image.LANCZOS)
        resized.append(im)
    total_w = sum(im.width for im in resized) + gap * (len(resized) - 1)
    canvas = Image.new("RGB", (total_w, target_h), (10, 10, 10))
    x = 0
    for im in resized:
        canvas.paste(im.convert("RGB"), (x, 0))
        x += im.width + gap
    return canvas


def _vstack(images: list[Image.Image], gap: int = 4) -> Image.Image:
    if not images:
        return Image.new("RGB", (1, 1))
    target_w = max(im.width for im in images)
    resized = []
    for im in images:
        if im.width != target_w:
            ratio = target_w / im.width
            im = im.resize((target_w, int(im.height * ratio)), Image.LANCZOS)
        resized.append(im)
    total_h = sum(im.height for im in resized) + gap * (len(resized) - 1)
    canvas = Image.new("RGB", (target_w, total_h), (10, 10, 10))
    y = 0
    for im in resized:
        canvas.paste(im.convert("RGB"), (0, y))
        y += im.height + gap
    return canvas


# ---------------------------------------------------------------------------
# Saliency overlay (pure-PIL fallback if matplotlib unavailable)
# ---------------------------------------------------------------------------

def _jet(v: float) -> tuple[int, int, int]:
    """Approximate 'jet' colormap for a scalar in [0, 1]."""
    v = float(np.clip(v, 0.0, 1.0))
    r = float(np.clip(1.5 - abs(4 * v - 3), 0, 1))
    g = float(np.clip(1.5 - abs(4 * v - 2), 0, 1))
    b = float(np.clip(1.5 - abs(4 * v - 1), 0, 1))
    return (int(r * 255), int(g * 255), int(b * 255))


def _saliency_overlay_pil(
    base: Image.Image,
    saliency: np.ndarray,
    alpha: float = 0.55,
) -> Image.Image:
    """Blend a jet heatmap onto *base* without requiring matplotlib."""
    try:
        from aba_rfdetr.saliency import overlay_saliency
        return overlay_saliency(base, saliency, colormap="jet", alpha=alpha).convert("RGB")
    except ImportError:
        pass

    # Fallback: vectorised numpy path
    h, w = saliency.shape
    # Build RGBA heatmap via numpy vectorisation
    v = saliency.astype(np.float32)
    r = np.clip(1.5 - np.abs(4 * v - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * v - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * v - 1), 0, 1)
    a = (alpha * v)
    heat = np.stack([r, g, b, a], axis=-1)
    heat = (heat * 255).astype(np.uint8)
    heat_pil = Image.fromarray(heat, mode="RGBA")
    return Image.alpha_composite(base.convert("RGBA"), heat_pil).convert("RGB")


# ---------------------------------------------------------------------------
# Core test routine
# ---------------------------------------------------------------------------

def run_test(
    image_path: Path,
    out_dir: Path,
    # RISE params
    n_masks: int = 256,
    mask_res: int = 8,
    mask_density: float = 0.5,
    # LIME params
    lime_samples: int = 256,
    lime_segments: int = 80,
    lime_segmentation: str = "slic",
    lime_positive_only: bool = False,
    # General
    method: str = "both",          # "rise" | "lime" | "both"
    stage1_only: bool = False,
    show_plot: bool = False,
) -> None:
    run_rise = method in ("rise", "both")
    run_lime = method in ("lime", "both")

    print(f"\n{'='*60}")
    print(f"  Image  : {image_path.name}")
    print(f"  Method : {method}")
    if run_rise:
        print(f"  RISE   : n_masks={n_masks}, mask_res={mask_res}, density={mask_density}")
    if run_lime:
        print(f"  LIME   : n_samples={lime_samples}, n_segments={lime_segments},"
              f" segmentation={lime_segmentation}")
    print(f"{'='*60}")

    # ── Load image ──────────────────────────────────────────────────────────
    image = Image.open(image_path).convert("RGB")
    stem_dir = out_dir / image_path.stem
    stem_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Image size : {image.size}")

    # ── Import pipeline ──────────────────────────────────────────────────────
    try:
        from aba_rfdetr.inference import (
            predict_pil_image_staged,
            load_inference_config,
        )
        from aba_rfdetr.saliency import (
            SaliencyConfig,
            LimeConfig,
            compute_stage1_saliency,
            compute_stage2_saliency,
            compute_lime_stage1,
            compute_lime_stage2,
        )
    except ImportError as exc:
        print(f"\n[ERROR] Could not import aba_rfdetr: {exc}")
        print("  Run from the project root with the package installed.")
        sys.exit(1)

    sal_cfg = SaliencyConfig(
        n_masks=n_masks,
        mask_res=mask_res,
        mask_density=mask_density,
        normalise=True,
        seed=42,
    )
    lime_cfg = LimeConfig(
        n_samples=lime_samples,
        n_segments=lime_segments,
        segmentation=lime_segmentation,
        positive_only=lime_positive_only,
        normalise=True,
        seed=42,
    )

    # ── Stage 1 detection ───────────────────────────────────────────────────
    print("\n[Stage 1] Running detection…")
    t0 = time.perf_counter()
    staged = predict_pil_image_staged(image)
    print(f"  Targets found : {len(staged.stage1_detections)}  ({time.perf_counter()-t0:.2f}s)")

    if not staged.stage1_detections:
        print("  No targets detected — saving blank annotated image and exiting.")
        _draw_boxes(image, [], title="No detections").save(stem_dir / "detections.png")
        return

    # Save annotated full image
    det_img = _draw_boxes(image, staged.stage1_detections, title="Stage 1 — Targets")
    det_img.save(stem_dir / "detections.png")
    print(f"  Saved : {stem_dir / 'detections.png'}")

    # Thumbnail for the summary strip
    thumb_w = min(det_img.width, 800)
    thumb_h = int(det_img.height * thumb_w / det_img.width)
    summary_rows: list[Image.Image] = [
        _add_label_bar(det_img.resize((thumb_w, thumb_h), Image.LANCZOS),
                       "Full image — Stage 1 detections")
    ]

    # ── Stage 1 saliency ────────────────────────────────────────────────────
    cfg_raw = load_inference_config()
    crop_padding = float(cfg_raw.get("stage2", {}).get("crop_padding", 0.10))

    for n_idx, s1_det in enumerate(staged.stage1_detections):
        print(f"\n[Stage 1 Saliency] Target {n_idx+1}/{len(staged.stage1_detections)}"
              f"  score={s1_det.score:.3f}  box={[int(v) for v in s1_det.box_xyxy]}")

        row_panels: list[Image.Image] = []

        # RISE
        if run_rise:
            t0 = time.perf_counter()
            sal_rise = compute_stage1_saliency(image, s1_det, cfg=sal_cfg)
            elapsed = time.perf_counter() - t0
            print(f"  RISE  {elapsed:.1f}s  |  "
                  f"min={sal_rise.min():.3f}  max={sal_rise.max():.3f}  mean={sal_rise.mean():.3f}")

            ov_rise = _saliency_overlay_pil(image, sal_rise)
            draw = ImageDraw.Draw(ov_rise)
            x1, y1, x2, y2 = [int(v) for v in s1_det.box_xyxy]
            draw.rectangle([x1, y1, x2, y2], outline=(255, 80, 80), width=3)

            fname = stem_dir / f"stage1_target_{n_idx}_rise.png"
            ov_rise.save(fname)
            print(f"  Saved : {fname}")

            tw = min(ov_rise.width, 600)
            th = int(ov_rise.height * tw / ov_rise.width)
            row_panels.append(
                _add_label_bar(ov_rise.resize((tw, th), Image.LANCZOS),
                               f"RISE — Target {n_idx}  (score {s1_det.score:.2f})"))

        # LIME
        if run_lime:
            t0 = time.perf_counter()
            try:
                sal_lime = compute_lime_stage1(image, s1_det, cfg=lime_cfg)
                elapsed = time.perf_counter() - t0
                print(f"  LIME  {elapsed:.1f}s  |  "
                      f"min={sal_lime.min():.3f}  max={sal_lime.max():.3f}  mean={sal_lime.mean():.3f}")

                ov_lime = _saliency_overlay_pil(image, sal_lime)
                draw = ImageDraw.Draw(ov_lime)
                x1, y1, x2, y2 = [int(v) for v in s1_det.box_xyxy]
                draw.rectangle([x1, y1, x2, y2], outline=(255, 80, 80), width=3)

                fname = stem_dir / f"stage1_target_{n_idx}_lime.png"
                ov_lime.save(fname)
                print(f"  Saved : {fname}")

                tw = min(ov_lime.width, 600)
                th = int(ov_lime.height * tw / ov_lime.width)
                row_panels.append(
                    _add_label_bar(ov_lime.resize((tw, th), Image.LANCZOS),
                                   f"LIME — Target {n_idx}  (score {s1_det.score:.2f})"))
            except ImportError as e:
                print(f"  LIME skipped: {e}")

        if row_panels:
            summary_rows.append(_hstack(row_panels, gap=6))

    # ── Stage 2 detection + saliency ────────────────────────────────────────
    if stage1_only:
        print("\n[Stage 2] Skipped (--stage1_only).")
    else:
        import base64
        import io as _io

        for crop_res in staged.crops:
            n_idx = crop_res.crop_index
            cx1, cy1 = int(crop_res.crop_box_xyxy[0]), int(crop_res.crop_box_xyxy[1])

            crop_bytes = base64.b64decode(crop_res.crop_image_base64)
            crop_pil = Image.open(_io.BytesIO(crop_bytes)).convert("RGB")

            print(f"\n[Stage 2] Crop {n_idx}  size={crop_pil.size}"
                  f"  detections={len(crop_res.detections)}")

            # Annotated crop (all bullets)
            crop_annot = _draw_boxes(crop_pil, crop_res.detections,
                                     title=f"Crop {n_idx} — Stage 2")
            crop_fname = stem_dir / f"stage2_crop_{n_idx}.png"
            crop_annot.save(crop_fname)
            print(f"  Saved : {crop_fname}")
            summary_rows.append(
                _add_label_bar(crop_annot, f"Stage 2 crop {n_idx} — all detections"))

            if not crop_res.detections:
                print("  No bullets detected in this crop.")
                continue

            for d_idx, s2_det in enumerate(crop_res.detections):
                print(f"  [{d_idx+1}/{len(crop_res.detections)}] "
                      f"{s2_det.label}  score={s2_det.score:.3f}  "
                      f"box={[int(v) for v in s2_det.box_xyxy]}")

                safe_label = s2_det.label.replace("/", "_")
                row_panels: list[Image.Image] = []

                # RISE
                if run_rise:
                    t0 = time.perf_counter()
                    sal = compute_stage2_saliency(crop_pil, s2_det, cfg=sal_cfg)
                    elapsed = time.perf_counter() - t0
                    print(f"    RISE {elapsed:.1f}s  |  "
                          f"min={sal.min():.3f}  max={sal.max():.3f}  mean={sal.mean():.3f}")

                    ov = _saliency_overlay_pil(crop_pil, sal)
                    draw = ImageDraw.Draw(ov)
                    bx1, by1, bx2, by2 = [int(v) for v in s2_det.box_xyxy]
                    draw.rectangle([bx1, by1, bx2, by2], outline=_colour(s2_det.label), width=2)

                    fname = stem_dir / f"stage2_{safe_label}_{n_idx}_{d_idx}_rise.png"
                    ov.save(fname)
                    print(f"    Saved : {fname}")

                    tw = min(ov.width, 400)
                    th = int(ov.height * tw / ov.width)
                    row_panels.append(
                        _add_label_bar(ov.resize((tw, th), Image.LANCZOS),
                                       f"RISE — {s2_det.label}  crop {n_idx}"
                                       f"  (score {s2_det.score:.2f})"))

                # LIME  — use fewer segments for small crops
                if run_lime:
                    crop_lime_cfg = LimeConfig(
                        n_samples=lime_cfg.n_samples,
                        n_segments=min(lime_cfg.n_segments, 40),
                        segmentation=lime_cfg.segmentation,
                        positive_only=lime_cfg.positive_only,
                        normalise=True,
                        seed=42,
                    )
                    t0 = time.perf_counter()
                    try:
                        sal_lime = compute_lime_stage2(crop_pil, s2_det, cfg=crop_lime_cfg)
                        elapsed = time.perf_counter() - t0
                        print(f"    LIME {elapsed:.1f}s  |  "
                              f"min={sal_lime.min():.3f}  max={sal_lime.max():.3f}"
                              f"  mean={sal_lime.mean():.3f}")

                        ov_lime = _saliency_overlay_pil(crop_pil, sal_lime)
                        draw = ImageDraw.Draw(ov_lime)
                        bx1, by1, bx2, by2 = [int(v) for v in s2_det.box_xyxy]
                        draw.rectangle([bx1, by1, bx2, by2],
                                       outline=_colour(s2_det.label), width=2)

                        fname = stem_dir / f"stage2_{safe_label}_{n_idx}_{d_idx}_lime.png"
                        ov_lime.save(fname)
                        print(f"    Saved : {fname}")

                        tw = min(ov_lime.width, 400)
                        th = int(ov_lime.height * tw / ov_lime.width)
                        row_panels.append(
                            _add_label_bar(ov_lime.resize((tw, th), Image.LANCZOS),
                                           f"LIME — {s2_det.label}  crop {n_idx}"
                                           f"  (score {s2_det.score:.2f})"))
                    except ImportError as e:
                        print(f"    LIME skipped: {e}")

                if row_panels:
                    summary_rows.append(_hstack(row_panels, gap=6))

    # ── Summary strip ────────────────────────────────────────────────────────
    if len(summary_rows) > 1:
        summary = _vstack(summary_rows, gap=6)
        summary_path = stem_dir / "summary.png"
        summary.save(summary_path)
        print(f"\n  Summary saved : {summary_path}")

        if show_plot:
            try:
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(14, max(4, len(summary_rows) * 3)))
                ax.imshow(np.asarray(summary))
                ax.axis("off")
                ax.set_title(f"Saliency Test — {image_path.name}", fontsize=14, pad=10)
                plt.tight_layout()
                plt.show()
            except ImportError:
                print("  (matplotlib not available — skipping interactive plot)")

    # ── JSON report ──────────────────────────────────────────────────────────
    report = {
        "image": str(image_path),
        "image_size": list(image.size),
        "method": method,
        "rise": {"n_masks": n_masks, "mask_res": mask_res, "mask_density": mask_density},
        "lime": {"n_samples": lime_samples, "n_segments": lime_segments,
                 "segmentation": lime_segmentation},
        "stage1_detections": [
            {
                "label": d.label,
                "score": round(d.score, 4),
                "box_xyxy": [round(v, 1) for v in d.box_xyxy],
            }
            for d in staged.stage1_detections
        ],
        "stage2_crops": [
            {
                "crop_index": cr.crop_index,
                "crop_box_xyxy": [round(v, 1) for v in cr.crop_box_xyxy],
                "detections": [
                    {
                        "label": d.label,
                        "score": round(d.score, 4),
                        "box_xyxy": [round(v, 1) for v in d.box_xyxy],
                    }
                    for d in cr.detections
                ],
            }
            for cr in staged.crops
        ],
    }
    report_path = stem_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"  Report saved  : {report_path}")
    print(f"\n  All outputs → {stem_dir}/\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Visual test for RF-DETR cascade + RISE/LIME saliency maps.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--image",     type=Path, help="Path to a single test image.")
    src.add_argument("--image_dir", type=Path, help="Directory of images to process.")

    p.add_argument("--out_dir", type=Path, default=Path("saliency_output"),
                   help="Root output directory (default: ./saliency_output/).")
    p.add_argument("--method", choices=["rise", "lime", "both"], default="both",
                   help="Which attribution method(s) to run (default: both).")

    # RISE
    rise = p.add_argument_group("RISE")
    rise.add_argument("--n_masks",      type=int,   default=256,
                      help="Number of random masks (default 256).")
    rise.add_argument("--mask_res",     type=int,   default=8,
                      help="Low-res grid size for mask generation (default 8).")
    rise.add_argument("--mask_density", type=float, default=0.5,
                      help="Fraction of pixels kept per mask (default 0.5).")

    # LIME
    lime = p.add_argument_group("LIME")
    lime.add_argument("--lime_samples",      type=int,   default=256,
                      help="Number of perturbed samples (default 256).")
    lime.add_argument("--lime_segments",     type=int,   default=80,
                      help="Target number of superpixel segments (default 80).")
    lime.add_argument("--lime_segmentation", choices=["slic", "quickshift"], default="slic",
                      help="Superpixel algorithm (default: slic).")
    lime.add_argument("--lime_positive_only", action="store_true",
                      help="Zero out negative LIME coefficients (show only supporting regions).")

    p.add_argument("--stage1_only", action="store_true",
                   help="Only compute Stage 1 (Target) saliency, skip Stage 2.")
    p.add_argument("--plot", action="store_true",
                   help="Display an interactive matplotlib figure after saving.")
    return p


def main() -> None:
    args = _build_parser().parse_args()

    image_paths: list[Path] = []
    if args.image:
        if not args.image.exists():
            print(f"[ERROR] Image not found: {args.image}")
            sys.exit(1)
        image_paths = [args.image]
    else:
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
        image_paths = sorted(
            p for p in args.image_dir.iterdir() if p.suffix.lower() in exts
        )
        if not image_paths:
            print(f"[ERROR] No images found in {args.image_dir}")
            sys.exit(1)
        print(f"Found {len(image_paths)} image(s) in {args.image_dir}")

    for img_path in image_paths:
        run_test(
            image_path=img_path,
            out_dir=args.out_dir,
            n_masks=args.n_masks,
            mask_res=args.mask_res,
            mask_density=args.mask_density,
            lime_samples=args.lime_samples,
            lime_segments=args.lime_segments,
            lime_segmentation=args.lime_segmentation,
            lime_positive_only=args.lime_positive_only,
            method=args.method,
            stage1_only=args.stage1_only,
            show_plot=args.plot,
        )

    print("Done.")


if __name__ == "__main__":
    main()
