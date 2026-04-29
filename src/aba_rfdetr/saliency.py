"""Saliency map generation for the two-stage RF-DETR cascade pipeline.

This module is copied from `XAI/saliency.py` so it can be imported as
`aba_rfdetr.saliency` by scripts such as `XAI/test_saliency.py`.
"""

from __future__ import annotations

# Re-export everything from the XAI implementation.
#
# Keeping the implementation in one place avoids divergence; the XAI scripts
# expect `aba_rfdetr.saliency` to exist.
from XAI.saliency import *  # noqa: F403

