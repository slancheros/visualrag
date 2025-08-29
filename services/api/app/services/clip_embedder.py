from typing import List
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from app.config import settings

class ClipEmbedder:
    _model = None
    _proc = None

    @classmethod
    def _load(cls):
        if cls._model is None:
            cls._model = CLIPModel.from_pretrained(settings.EMBEDDING_MODEL)
            cls._model.eval()
        if cls._proc is None:
            cls._proc = CLIPProcessor.from_pretrained(settings.EMBEDDING_MODEL)

    @classmethod
    def image_embedding(cls, img: Image.Image) -> List[float]:
        cls._load()
        inputs = cls._proc(images=img, return_tensors="pt")
        with torch.no_grad():
            feats = cls._model.get_image_features(**inputs)
        feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
        return feats.squeeze(0).cpu().tolist()
