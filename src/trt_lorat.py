"""
TensorRT runtime wrapper for the LoRAT engines (see export_lorat_trt.py).

Loads the FP16 .engine and exposes a callable with the same signature and return
contract as `LoRAT_DINOv2.forward` — `(z=, x=, z_feat_mask=) -> {'score_map',
'boxes'}` — so lorat_infer.LoRATTracker can swap it in for `self.model` and every
other step (SiamFC cropping, the box_with_score_map post-process, the search
region provider) stays exactly as it is in the torch path.

I/O binds directly to torch CUDA tensors via data_ptr() (no extra copies) and the
engine runs on the current torch CUDA stream. Engine I/O is FP32 (plus INT32 for
the mask); the compute inside is FP16.
"""
import os

import torch
import tensorrt as trt

import os
# Every path is resolved from where THIS repo was cloned, so it works wherever it
# lands. setup_model.sh writes the same values into the launcher scripts.
ROOT = os.environ.get("REPO_ROOT",
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ENGINE_DIR = os.environ.get("ENGINE_DIR", f"{ROOT}/weights/lorat_trt")

_TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# trt.nptype() is unusable here: TRT 8.5 references the removed np.bool alias.
_TRT_TO_TORCH = {
    trt.DataType.FLOAT: torch.float32,
    trt.DataType.HALF: torch.float16,
    trt.DataType.INT32: torch.int32,
    trt.DataType.INT8: torch.int8,
    trt.DataType.BOOL: torch.bool,
}


# Engine precisions the naming scheme understands. The precision is part of the
# tracker key (lorat_l224_trt_fp16, lorat_l224_trt_int8) so a result directory can
# never leave it ambiguous which engine produced the numbers.
PRECISIONS = ("fp16", "int8")


def engine_path(variant, precision="fp16"):
    if precision not in PRECISIONS:
        raise ValueError(f"unknown engine precision {precision!r}; "
                         f"expected one of {PRECISIONS}")
    return os.path.join(ENGINE_DIR, f"{variant}_{precision}.engine")


def available(variant, precision="fp16"):
    return os.path.isfile(engine_path(variant, precision))


class TRTLoRATModel:
    """Minimal TRT executor with a LoRAT_DINOv2-compatible __call__."""

    IN_NAMES = ("z", "x", "z_feat_mask")
    OUT_NAMES = ("score_map", "boxes")

    def __init__(self, variant, path=None, device="cuda", precision="fp16"):
        path = path or engine_path(variant, precision)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"engine not found: {path}\n"
                f"build it with: venvs/lorat/bin/python "
                f"benchmark/scripts/export_lorat_trt.py --variant {variant}")
        self.variant = variant
        self.precision = precision
        self.path = path
        self.device = torch.device(device)
        with open(path, "rb") as f, trt.Runtime(_TRT_LOGGER) as rt:
            self.engine = rt.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        self.bindings = {}
        for i in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            self.bindings[name] = dict(
                index=i,
                is_input=self.engine.binding_is_input(i),
                shape=tuple(self.engine.get_binding_shape(i)),
                dtype=_TRT_TO_TORCH[self.engine.get_binding_dtype(i)])
        missing = [n for n in self.IN_NAMES + self.OUT_NAMES if n not in self.bindings]
        if missing:
            raise RuntimeError(f"engine {path} is missing bindings {missing}")

        # static shapes -> allocate the outputs once
        self._outputs = {n: torch.empty(self.bindings[n]["shape"],
                                        dtype=self.bindings[n]["dtype"],
                                        device=self.device)
                         for n in self.OUT_NAMES}

    def _prep(self, name, t):
        b = self.bindings[name]
        t = t.to(self.device, b["dtype"]).contiguous()
        if tuple(t.shape) != b["shape"]:
            raise ValueError(f"{name}: engine expects {b['shape']}, got {tuple(t.shape)}")
        return t

    def __call__(self, z, x, z_feat_mask):
        z = self._prep("z", z)
        x = self._prep("x", x)
        m = self._prep("z_feat_mask", z_feat_mask)

        ptrs = [0] * self.engine.num_bindings
        for name, t in (("z", z), ("x", x), ("z_feat_mask", m)):
            ptrs[self.bindings[name]["index"]] = t.data_ptr()
        for n in self.OUT_NAMES:
            ptrs[self.bindings[n]["index"]] = self._outputs[n].data_ptr()

        stream = torch.cuda.current_stream(self.device)
        self.context.execute_async_v2(ptrs, stream.cuda_stream)
        stream.synchronize()
        # clone so the next call's in-place engine write cannot alias what the
        # post-process is still holding
        return {n: self._outputs[n].clone() for n in self.OUT_NAMES}

    forward = __call__

    def eval(self):      # torch-module compatibility, no-op
        return self
