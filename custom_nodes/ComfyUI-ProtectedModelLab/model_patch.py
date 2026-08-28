from __future__ import annotations

from comfy.weight_adapter.lokr import LoKrAdapter
from comfy.weight_adapter.lora import LoRAAdapter

from .core_loader import ProtectedCoreHandle


def apply_core(model, handle: ProtectedCoreHandle, strength: float):
    patched = model.clone()
    state_keys = set(patched.model.state_dict().keys())
    patches = {}
    missing = []
    for layer in handle.core.layers:
        model_key = "diffusion_model." + layer.model_key
        if model_key not in state_keys:
            missing.append(model_key)
            continue
        if layer.adapter_type == "lokr":
            weights = (layer.w1, layer.w2 * layer.layer_scale, None, None, None, None, None, None, None)
            patches[model_key] = LoKrAdapter(set(), weights)
        elif layer.adapter_type == "lora":
            rank = layer.w2.shape[0]
            weights = (layer.w1 * layer.layer_scale, layer.w2, float(rank), None, None, None)
            patches[model_key] = LoRAAdapter(set(), weights)
        else:
            raise ValueError(f"unsupported protected adapter type: {layer.adapter_type}")
    if missing:
        preview = "\n".join(missing[:20])
        raise ValueError(f"protected core has {len(missing)} unmatched model tensors:\n{preview}")
    applied = patched.add_patches(patches, strength_patch=strength)
    if len(applied) != len(patches):
        raise ValueError(f"ModelPatcher accepted {len(applied)}/{len(patches)} protected tensors")
    return patched
