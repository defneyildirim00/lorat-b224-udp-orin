"""
Standalone, faithful LoRAT inference wrapper for the Jetson AGX Orin.

LoRAT (ECCV 2024, LitingLin/LoRAT) ships a full training/eval framework
(`trackit`) whose evaluation path is distributed/batched/cache-based and not
usable interactively. This module drives the SAME official code with a minimal
single-stream init/track loop:

  * model built by the official `build_LoRAT_model` (DINOv2 backbone from
    facebook public weights, LoRA merged at inference via the repo's own
    state-dict hooks);
  * the exact official pre/post-processing, imported (never reimplemented):
      - template/search SiamFC cropping .......... trackit.core.utils.siamfc_cropping
      - template foreground token mask ........... one_stream plugin's get_foreground_bounding_box
      - box_with_score_map post-process .......... components/post_process/box_with_score_map
      - search-region box memory ................. SiamFCCroppingParameterSimpleProvider
    reproducing one_stream/__init__.py::track() step for step.

The `trackit` codebase uses PEP 604 (`X | None`) in runtime-evaluated
annotations, which raises TypeError on this Jetson's Python 3.8. We apply the
behavior-preserving PEP 563 fix (`from __future__ import annotations`) to the
specific files that get imported, discovered by catching the TypeError and
patching the offending file from the traceback. This changes no logic.

Variant params (from config/LoRAT/{B,L}-{224,378}) are hard-coded below so we do
not need the framework's yaml/boot machinery.
"""
from __future__ import annotations
import os
import sys
import ast
import traceback
from typing import Optional, Tuple

import numpy as np
import torch

LORAT_ROOT = os.environ.get(
    "LORAT_ROOT",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "third_party", "LoRAT"))

# ------------------------------------------------------------------ variants
# common to all: template_area_factor=2.0, interp=bilinear/align_corners=False,
# normalization=imagenet, lora r=64 alpha=64, post-process window_penalty=0.45,
# search-region provider min_object_size=10.
_COMMON = dict(
    template_area_factor=2.0,
    interpolation_mode="bilinear",
    interpolation_align_corners=False,
    normalization="imagenet",
    window_penalty=0.45,
    min_object_size=10,
    lora=dict(r=64, alpha=64, dropout=0.0, use_rslora=False),
)

# ------------------------------------------------------------------- motion
# Optional score-map prior (motion=True). The official post-process penalises the
# score map with a Hann window centred on the search crop, i.e. on the target's
# LAST position. With a fast, small target among look-alikes that prior votes for
# whichever distractor sits closest to where the target used to be. MOTION swaps
# that window for a Gaussian centred on an EMA velocity extrapolation of where
# the target should be NOW, so candidates off the motion path are suppressed
# instead. Nothing else changes: same model, same cropping, same box decoding.
_MOTION_VEL_EMA = 0.7      # velocity smoothing: v <- 0.7*v + 0.3*(c_t - c_{t-1})
_MOTION_SIGMA_DIV = 4.0    # Gaussian sigma = response_map_height / 4

VARIANTS = {
    "lorat_b224": dict(backbone="ViT-B/14", drop_path_rate=0.0,
                       template_size=(112, 112), template_feat_size=(8, 8),
                       search_region_size=(224, 224), search_region_feat_size=(16, 16),
                       search_area_factor=4.0),
    "lorat_b378": dict(backbone="ViT-B/14", drop_path_rate=0.0,
                       template_size=(196, 196), template_feat_size=(14, 14),
                       search_region_size=(378, 378), search_region_feat_size=(27, 27),
                       search_area_factor=5.0),
    "lorat_l224": dict(backbone="ViT-L/14", drop_path_rate=0.1,
                       template_size=(112, 112), template_feat_size=(8, 8),
                       search_region_size=(224, 224), search_region_feat_size=(16, 16),
                       search_area_factor=4.0),
    "lorat_l378": dict(backbone="ViT-L/14", drop_path_rate=0.1,
                       template_size=(196, 196), template_feat_size=(14, 14),
                       search_region_size=(378, 378), search_region_feat_size=(27, 27),
                       search_area_factor=5.0),
}


def _build_config(variant: str) -> dict:
    v = VARIANTS[variant]
    backbone_params = {"name": v["backbone"], "acc": "default"}
    if v["drop_path_rate"] > 0:
        backbone_params["drop_path_rate"] = v["drop_path_rate"]
    return {
        "common": {
            "template_size": list(v["template_size"]),
            "search_region_size": list(v["search_region_size"]),
            "template_feat_size": list(v["template_feat_size"]),
            "search_region_feat_size": list(v["search_region_feat_size"]),
            "response_map_size": list(v["search_region_feat_size"]),
            "interpolation_mode": _COMMON["interpolation_mode"],
            "interpolation_align_corners": _COMMON["interpolation_align_corners"],
            "normalization": _COMMON["normalization"],
        },
        "model": {
            "type": "dinov2",
            "backbone": {"type": "DINOv2", "parameters": backbone_params},
            "lora": dict(_COMMON["lora"]),
        },
    }


# ------------------------------------------------------- PEP 604 auto-patcher
def _add_future_import(path: str) -> bool:
    """Insert `from __future__ import annotations` after the module docstring.
    Returns True if the file was modified."""
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    if "from __future__ import annotations" in src:
        return False
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return False
    insert_line = 0  # 0-based index into lines
    if (tree.body and isinstance(tree.body[0], ast.Expr)
            and isinstance(getattr(tree.body[0], "value", None), ast.Constant)
            and isinstance(tree.body[0].value.value, str)):
        insert_line = tree.body[0].end_lineno  # after the docstring
    lines = src.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    lines.insert(insert_line, "from __future__ import annotations\n")
    with open(path, "w", encoding="utf-8") as f:
        f.write("".join(lines))
    return True


def _purge_trackit():
    for name in list(sys.modules):
        if name == "trackit" or name.startswith("trackit."):
            del sys.modules[name]


# TypeError messages raised on Python 3.8 when a module evaluates a modern
# annotation at import time — all fixed by `from __future__ import annotations`:
#   PEP 604  x: int | None      -> "unsupported operand type(s) for |"
#   PEP 585  x: tuple[int, ...] -> "'type' object is not subscriptable"
_ANNOTATION_ERRORS = ("unsupported operand type", "object is not subscriptable")


def _run_with_autopatch(fn, max_iter: int = 400):
    """Run fn(); on a Py3.8 annotation TypeError, patch the offending trackit
    file with `from __future__ import annotations` and retry."""
    if LORAT_ROOT not in sys.path:
        sys.path.insert(0, LORAT_ROOT)
    last = None
    for _ in range(max_iter):
        try:
            return fn()
        except (TypeError, AttributeError, NameError) as e:
            # TypeError signatures are annotation-only; AttributeError/NameError
            # (e.g. `Optional[dist.Backend]` when torch.distributed is disabled,
            # or a forward-referenced type) are only patched when the offending
            # trackit line is actually an annotation. _add_future_import is
            # idempotent (returns False if already patched) -> no infinite loop
            # and a genuine runtime error surfaces on the next pass.
            if isinstance(e, TypeError) and not any(sig in str(e) for sig in _ANNOTATION_ERRORS):
                raise
            tb = traceback.extract_tb(e.__traceback__)
            target = None
            for fr in reversed(tb):
                if fr.filename.startswith(LORAT_ROOT) and fr.filename.endswith(".py"):
                    line = fr.line or ""
                    if not isinstance(e, TypeError) and ("->" not in line and ":" not in line):
                        continue  # not an annotation line — don't patch
                    target = fr.filename
                    break
            if target is None or not _add_future_import(target):
                raise
            last = target
            _purge_trackit()
    raise RuntimeError(f"autopatch exceeded max_iter (last patched: {last})")


# ------------------------------------------------------------------ tracker
class LoRATTracker:
    def __init__(self, variant: str, weight_path: Optional[str] = None,
                 device: str = "cuda", amp: bool = True, fp16_weights: bool = True,
                 engine: bool = False, motion: bool = False,
                 engine_precision: str = "fp16"):
        """amp=True runs the backbone in fp16.

        motion=True replaces the post-process Hann window with a velocity-centred
        Gaussian (see _MOTION_* above). Post-processing only — no effect on the
        model, so it composes with engine=True. Default False keeps the official
        behaviour bit for bit.

        engine=True replaces the torch model with the TensorRT FP16 engine built
        by export_lorat_trt.py. Only the model forward is swapped — the SiamFC
        cropping, the foreground token mask, the box_with_score_map post-process
        and the search-region provider all stay on the official torch code path.

        fp16_weights=True casts the weights to fp16 ONCE at build time instead of
        wrapping every forward in torch.autocast. autocast caches its fp32->fp16
        weight casts only for the lifetime of the context, so entering and
        leaving it per frame re-cast all 88M/308M parameters on every single
        frame. Worth 25% of B-224's forward and 15% of L-378's, which comes out
        as 1.13-1.22x end to end once cropping and post-processing are counted.
        The difference in numerics is that layernorm/softmax also run in fp16;
        verify_lorat_fp16.py checks that against the autocast path (mean IoU
        0.998-0.999 between the two, accuracy vs GT unchanged).
        """
        assert variant in VARIANTS, f"unknown LoRAT variant {variant!r}; choose {list(VARIANTS)}"
        self.variant = variant
        self.cfg = _build_config(variant)
        self.device = torch.device(device)
        self.engine = engine
        self.engine_precision = engine_precision
        self.motion = motion
        self.amp = amp and self.device.type == "cuda" and not engine
        # the engine is already fp16 internally; its bindings are fp32
        self.fp16_weights = self.amp and fp16_weights
        self._v = VARIANTS[variant]

        common = self.cfg["common"]
        self.template_size = np.array(common["template_size"])          # (W, H)
        self.search_size = np.array(common["search_region_size"])       # (W, H)
        self.template_feat = tuple(common["template_feat_size"])        # (W, H)
        self.interp = common["interpolation_mode"]
        self.align = common["interpolation_align_corners"]
        self.template_area_factor = _COMMON["template_area_factor"]
        self.search_area_factor = self._v["search_area_factor"]

        _run_with_autopatch(lambda: self._build(weight_path))

    # -- build model + import official utilities (under autopatch) -----------
    def _build(self, weight_path):
        from trackit.models import ModelImplementationSuggestions
        from trackit.models.methods.LoRAT.builder import build_LoRAT_model
        from trackit.core.utils.siamfc_cropping import (
            get_siamfc_cropping_params, apply_siamfc_cropping,
            apply_siamfc_cropping_to_boxes, reverse_siamfc_cropping_params)
        from trackit.core.transforms.dataset_norm_stats import get_dataset_norm_stats_transform
        from trackit.core.operator.numpy.bbox.utility.image import bbox_clip_to_image_boundary_
        from trackit.runners.evaluation.common.siamfc_search_region_cropping_params_provider.simple import (
            SiamFCCroppingParameterSimpleProvider)
        from trackit.runners.evaluation.distributed.tracker_evaluator.components.post_process.box_with_score_map import (
            PostProcessing_BoxWithScoreMap)
        from trackit.runners.evaluation.distributed.tracker_evaluator.default.pipelines.utils.bbox_mask_gen import (
            get_foreground_bounding_box)

        self._f = dict(
            get_siamfc_cropping_params=get_siamfc_cropping_params,
            apply_siamfc_cropping=apply_siamfc_cropping,
            apply_siamfc_cropping_to_boxes=apply_siamfc_cropping_to_boxes,
            reverse_siamfc_cropping_params=reverse_siamfc_cropping_params,
            clip_=bbox_clip_to_image_boundary_,
            get_foreground_bounding_box=get_foreground_bounding_box,
        )
        self._Provider = SiamFCCroppingParameterSimpleProvider

        if self.engine:
            # The engine already carries the merged LoRA weights, so skip
            # building the torch model altogether — that would otherwise pull the
            # DINOv2 backbone (1.2 GB for ViT-L) off disk and construct 308 M
            # parameters only to throw them away. Everything else above and below
            # (cropping, mask, post-process, provider) is unchanged.
            from trt_lorat import TRTLoRATModel
            self.model = TRTLoRATModel(self.variant, device=str(self.device),
                                       precision=self.engine_precision)
            print(f"[LoRAT:{self.variant}] using TensorRT engine "
                  f"{os.path.basename(self.model.path)}")
        else:
            # model (fp32 weights; fp16 compute via autocast or a one-off .half()).
            # load_pretrained pulls the DINOv2 base weights; optimize_for_inference
            # merges LoRA.
            suggestions = ModelImplementationSuggestions(
                device=self.device, dtype=torch.float32,
                optimize_for_inference=True, load_pretrained=True)
            model = build_LoRAT_model(self.cfg, suggestions)
            model.eval().to(self.device)

            if weight_path is not None:
                self._load_weight(model, weight_path)
            if self.fp16_weights:
                # cast once, here — not once per frame inside autocast
                model.half()
            self.model = model

        self.norm_ = get_dataset_norm_stats_transform(self.cfg["common"]["normalization"], inplace=True)
        self.post_process = PostProcessing_BoxWithScoreMap(
            self.device,
            response_map_size=tuple(self.cfg["common"]["response_map_size"]),
            search_region_size=tuple(self.cfg["common"]["search_region_size"]),
            window_penalty_ratio=_COMMON["window_penalty"])
        self.post_process.start()

    @staticmethod
    def _is_safetensors(path):
        # safetensors = 8-byte little-endian header length followed by a JSON '{'.
        # The released LoRAT weights are safetensors despite their .bin extension.
        if path.endswith(".safetensors"):
            return True
        try:
            with open(path, "rb") as fh:
                head = fh.read(9)
            n = int.from_bytes(head[:8], "little")
            return 0 < n < (1 << 30) and head[8:9] == b"{"
        except Exception:
            return False

    def _load_weight(self, model, weight_path):
        from trackit.core.runtime.utils.checkpoint.load import load_model_weight
        use_st = self._is_safetensors(weight_path)
        result = load_model_weight(model, weight_path, use_safetensors=use_st, strict=False)
        if result is not None:
            missing, unexpected = result
            # base ViT weights come from load_pretrained; the checkpoint provides
            # lora + head + token_type_embed. pos_embed is EXPECTED-missing: it is
            # derived (interpolated) from the DINOv2 backbone in the model ctor and
            # frozen during LoRA training, so the checkpoint omits it. Only warn on
            # genuinely trained params that failed to load.
            head_missing = [k for k in missing if k.startswith(("head.", "token_type_embed"))]
            if head_missing:
                print(f"[LoRAT:{self.variant}] WARNING missing head/embed keys: {head_missing[:6]}"
                      f"{' ...' if len(head_missing) > 6 else ''}")
            if unexpected:
                print(f"[LoRAT:{self.variant}] {len(unexpected)} unexpected checkpoint keys "
                      f"(e.g. {list(unexpected)[:3]})")
        print(f"[LoRAT:{self.variant}] loaded weights from {os.path.basename(weight_path)}")

    # -- public API ----------------------------------------------------------
    @torch.inference_mode()
    def initialize(self, image_rgb: np.ndarray, bbox_xyxy):
        """image_rgb: HxWx3 uint8 RGB. bbox_xyxy: (x1,y1,x2,y2) in image pixels."""
        f = self._f
        z_bbox = np.asarray(bbox_xyxy, dtype=np.float64)
        z_image = torch.from_numpy(np.ascontiguousarray(image_rgb)).permute(2, 0, 1).to(self.device).float()

        tpar = f["get_siamfc_cropping_params"](z_bbox, self.template_area_factor, self.template_size)
        z, z_mean, tpar = f["apply_siamfc_cropping"](
            z_image, self.template_size, tpar, self.interp, self.align)
        z = z / 255.0
        self.norm_(z)                           # cropping/normalization stay fp32
        self.z = z.unsqueeze(0)                 # (1,3,Ht,Wt)
        if self.fp16_weights:
            self.z = self.z.half()
        self.z_mean = z_mean                    # (3,)

        # template foreground token mask (official plugin math)
        feat_w, feat_h = self.template_feat
        stride = (self.template_size[0] / feat_w, self.template_size[1] / feat_h)
        fg = f["get_foreground_bounding_box"](z_bbox, tpar, stride)
        f["clip_"](fg, np.array((feat_w, feat_h)))
        mask = torch.zeros((feat_h, feat_w), dtype=torch.long)
        fg = torch.from_numpy(fg)
        mask[fg[1]:fg[3], fg[0]:fg[2]] = 1
        self.z_feat_mask = mask.unsqueeze(0).to(self.device)   # (1,feat_h,feat_w)

        self.provider = self._Provider(self.search_area_factor, min_object_size=_COMMON["min_object_size"])
        self.provider.initialize(z_bbox)
        self.last_box = z_bbox.copy()
        # motion state: last centre + smoothed velocity, both in full-image pixels
        self._prev_c = np.array([(z_bbox[0] + z_bbox[2]) / 2.0,
                                 (z_bbox[1] + z_bbox[3]) / 2.0])
        self._vel = np.zeros(2)

    @torch.inference_mode()
    def track(self, image_rgb: np.ndarray):
        """Returns (bbox_xyxy float64 (4,), confidence float)."""
        f = self._f
        x_image = torch.from_numpy(np.ascontiguousarray(image_rgb)).permute(2, 0, 1).to(self.device).float()
        H, W = x_image.shape[-2:]
        image_size = np.array((W, H), dtype=np.int32)

        cparams = self.provider.get(self.search_size)
        x, _, cparams = f["apply_siamfc_cropping"](
            x_image, self.search_size, cparams, self.interp, self.align, self.z_mean)
        x = x / 255.0
        self.norm_(x)
        x = x.unsqueeze(0)

        if self.engine:
            out = self.model(z=self.z, x=x, z_feat_mask=self.z_feat_mask)
        elif self.fp16_weights:
            out = self.model(z=self.z, x=x.half(), z_feat_mask=self.z_feat_mask)
        else:
            with torch.autocast(self.device.type, enabled=self.amp, dtype=torch.float16):
                out = self.model(z=self.z, x=x, z_feat_mask=self.z_feat_mask)
        if self.motion:
            self.post_process._window = self._motion_window(out["score_map"].shape, cparams)
        result = self.post_process(out)
        conf = float(result["confidence"][0].item())
        box = result["box"][0].detach().to(torch.float64).cpu().numpy()   # search-crop pixel coords

        box_full = f["apply_siamfc_cropping_to_boxes"](box, f["reverse_siamfc_cropping_params"](cparams))
        f["clip_"](box_full, image_size)
        self.provider.update(conf, box_full, image_size)
        self.last_box = box_full
        if self.motion:
            c = np.array([(box_full[0] + box_full[2]) / 2.0, (box_full[1] + box_full[3]) / 2.0])
            self._vel = _MOTION_VEL_EMA * self._vel + (1 - _MOTION_VEL_EMA) * (c - self._prev_c)
            self._prev_c = c
        return box_full, conf

    def aim(self, bbox_xyxy):
        """Move the search-region centre to an externally supplied box.

        The template (self.z) and the motion state are left untouched: this only
        rewrites the cropping provider's cached box, i.e. WHERE the next frame is
        searched, not what is being looked for. The SAMURAI cascade uses it to
        hand a recovered target back to LoRAT (see cascade_infer.py); a wrong aim
        is therefore recoverable — it costs search position, not the template.
        """
        if getattr(self, "provider", None) is None:
            raise RuntimeError("aim() before initialize()")
        self.provider.cached_bbox[:] = np.asarray(bbox_xyxy, dtype=np.float64)

    # -- motion prior ---------------------------------------------------------
    def _motion_window(self, score_map_shape, cparams):
        """Gaussian score-map prior centred on prev_centre + velocity, expressed
        in response-map cells. Replaces the Hann window for this frame only."""
        _, Fh, Fw = score_map_shape
        pred_c = self._prev_c + self._vel
        # a degenerate 2x2 box is enough to carry the point through the official
        # full-image -> search-crop mapping (same params the frame was cropped with)
        pb = np.array([pred_c[0] - 1, pred_c[1] - 1, pred_c[0] + 1, pred_c[1] + 1], dtype=np.float64)
        sr = self._f["apply_siamfc_cropping_to_boxes"](pb, cparams)
        cx, cy = (sr[0] + sr[2]) / 2.0, (sr[1] + sr[3]) / 2.0
        cc = float(np.clip(cx / (self.search_size[0] / Fw), 0, Fw - 1))
        cr = float(np.clip(cy / (self.search_size[1] / Fh), 0, Fh - 1))
        yy, xx = np.mgrid[0:Fh, 0:Fw]
        sig = Fh / _MOTION_SIGMA_DIV
        g = np.exp(-((yy - cr) ** 2 + (xx - cc) ** 2) / (2 * sig * sig))
        return torch.from_numpy(g.reshape(-1).astype(np.float32)).to(self.device)
