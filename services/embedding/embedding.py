# services/embedding/embedding.py
from __future__ import annotations
import os
from typing import List, Tuple
from PIL import Image

import torch
from transformers import CLIPModel, CLIPProcessor

# Carga perezosa y configurable por variables de entorno
_MODEL_NAME = os.getenv("EMB_MODEL_NAME", "openai/clip-vit-base-patch32")
_MODEL_DIR  = os.getenv("EMB_MODEL_DIR", None)

_clip_model: CLIPModel | None = None
_clip_proc: CLIPProcessor | None = None

def load_model() -> None:
    """Carga el modelo una sola vez."""
    global _clip_model, _clip_proc
    if _clip_model is not None:
        return

    kwargs = {}
    if _MODEL_DIR:
        kwargs["cache_dir"] = _MODEL_DIR

    # Usa pesos en safetensors cuando estén disponibles
    _clip_model = CLIPModel.from_pretrained(_MODEL_NAME, **kwargs)
    _clip_proc  = CLIPProcessor.from_pretrained(_MODEL_NAME, **kwargs)
    _clip_model.eval()  # inferencia

def embed_image(image_path: str) -> Tuple[List[float], int]:
    """Retorna (vector, dimension)."""
    global _clip_model, _clip_proc

    assert _clip_model is not None and _clip_proc is not None
    img = Image.open(image_path).convert("RGB")
    inputs = _clip_proc(images=img, return_tensors="pt")

    with torch.no_grad():
        feats = _clip_model.get_image_features(**inputs)
        feats = feats / feats.norm(p=2, dim=-1, keepdim=True)

    vec = feats.squeeze(0).tolist()
    return vec, len(vec)

