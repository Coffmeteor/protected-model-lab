from __future__ import annotations

import os
import re
from pathlib import Path

import comfy.sd
import folder_paths
import numpy as np
from PIL import Image

from common.hashing import sha256_file

from .core_loader import ProtectedIdentity, load_bound_core
from .model_patch import apply_core
from .validation import resolved_model_path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ProtectedCarrierLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"carrier_name": (folder_paths.get_filename_list("protected_carriers"),)}}

    RETURN_TYPES = ("MODEL", "PROTECTED_IDENTITY")
    RETURN_NAMES = ("model", "protected_identity")
    FUNCTION = "load_carrier"
    CATEGORY = "protected_model_lab"

    def load_carrier(self, carrier_name):
        path = resolved_model_path("protected_carriers", carrier_name, ".safetensors")
        identity = ProtectedIdentity(path, sha256_file(path), "krea2")
        model = comfy.sd.load_diffusion_model(str(path), model_options={})
        return model, identity


class ProtectedCoreLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "core_name": (folder_paths.get_filename_list("protected_cores"),),
                "protected_identity": ("PROTECTED_IDENTITY",),
            }
        }

    RETURN_TYPES = ("PROTECTED_CORE",)
    RETURN_NAMES = ("protected_core",)
    FUNCTION = "load_core"
    CATEGORY = "protected_model_lab"

    def load_core(self, core_name, protected_identity):
        path = resolved_model_path("protected_cores", core_name, ".spcore")
        return (load_bound_core(path, protected_identity),)


class ApplyProtectedCore:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "protected_core": ("PROTECTED_CORE",),
                "strength": ("FLOAT", {"default": 1.0, "min": -4.0, "max": 4.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "protected_model_lab"

    def apply(self, model, protected_core, strength):
        return (apply_core(model, protected_core, strength),)


class ProtectedValidationSave:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "variant": ("STRING", {"default": "teacher"}),
                "seed": ("INT", {"default": 812301, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "protected_model_lab"

    def save(self, images, variant, seed):
        if re.fullmatch(r"[a-z0-9_]{1,80}", variant) is None:
            raise ValueError("variant must contain only lowercase letters, digits, and underscores")
        output = PROJECT_ROOT / "outputs/validation_images"
        output.mkdir(parents=True, exist_ok=True)
        for index, image in enumerate(images):
            array = np.clip(image.cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
            destination = output / f"{variant}_s{seed}_{index:02d}.png"
            temp = destination.with_name(destination.name + ".tmp")
            Image.fromarray(array).save(temp, format="PNG", compress_level=4)
            os.replace(temp, destination)
        return ()


NODE_CLASS_MAPPINGS = {
    "ProtectedCarrierLoader": ProtectedCarrierLoader,
    "ProtectedCoreLoader": ProtectedCoreLoader,
    "ApplyProtectedCore": ApplyProtectedCore,
    "ProtectedValidationSave": ProtectedValidationSave,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ProtectedCarrierLoader": "Protected Carrier Loader",
    "ProtectedCoreLoader": "Protected Core Loader",
    "ApplyProtectedCore": "Apply Protected Core",
    "ProtectedValidationSave": "Protected Validation Save",
}
