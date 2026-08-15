"""Uniform tracker interface for track_live.py — LoRAT B-224 only.

The benchmark this was extracted from carries six tracker families behind one
API; this repo carries one, so everything about the others is gone. Boxes are
xywh at this boundary, whatever the model uses internally.
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("REPO_ROOT", os.path.dirname(_HERE))

# `_motion` swaps the post-process Hann window for a velocity-centred Gaussian
# (lorat_infer._MOTION_*): post-processing only, so it combines with the engine.
# The engine precision is part of the key — lorat_b224_trt_fp16, never a bare
# `_trt` — so a run can never leave it ambiguous which engine produced it.
BASE = "lorat_b224"
KEYS = {BASE, BASE + "_motion",
        BASE + "_trt_fp16", BASE + "_motion_trt_fp16",
        BASE + "_trt_int8", BASE + "_motion_trt_int8"}
_WEIGHT_CANDIDATES = ("base.bin", "LoRAT-B-224.bin", "B-224.bin", "lorat_b224.bin")


class UnifiedTracker:
    def __init__(self, impl):
        self._impl = impl

    def init(self, rgb, bbox_xywh):
        x, y, w, h = (float(v) for v in bbox_xywh)
        self._impl.initialize(rgb, np.array([x, y, x + w, y + h], dtype=np.float64))

    def track(self, rgb):
        """Returns (bbox_xywh list, score)."""
        xyxy, conf = self._impl.track(rgb)
        x1, y1, x2, y2 = (float(v) for v in xyxy)
        return [x1, y1, x2 - x1, y2 - y1], float(conf)


def _weight_path():
    for name in _WEIGHT_CANDIDATES:
        cand = os.path.join(ROOT, "weights", "lorat", name)
        if os.path.isfile(cand):
            return cand
    return None


def build(tracker_key, config=None, device="cuda", amp=True):
    """Return (UnifiedTracker, config_string)."""
    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)
    if tracker_key not in KEYS:
        raise SystemExit("unknown tracker %r; this repo has: %s"
                         % (tracker_key, ", ".join(sorted(KEYS))))

    from lorat_infer import LoRATTracker
    variant, use_engine, precision = tracker_key, False, "fp16"
    for p in ("fp16", "int8"):
        if variant.endswith("_trt_" + p):
            use_engine, precision = True, p
            variant = variant[: -len("_trt_" + p)]
            break
    use_motion = variant.endswith("_motion")
    if use_motion:
        variant = variant[: -len("_motion")]

    wp = _weight_path()
    impl = LoRATTracker(variant, weight_path=wp, device=device, amp=amp,
                        engine=use_engine, motion=use_motion,
                        engine_precision=precision)
    cfg = tracker_key if wp else tracker_key + " [NO WEIGHTS - random head]"
    return UnifiedTracker(impl), cfg
