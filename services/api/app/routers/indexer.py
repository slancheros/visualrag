from fastapi import APIRouter, UploadFile, File, Form
from typing import Optional
from uuid import uuid4
from io import BytesIO
import os, requests
from PIL import Image
from app.config import settings
from app.services.clip_embedder import ClipEmbedder
from app.services.weaviate_client import ensure_schema, upsert_item

router = APIRouter()



def _save_local(file_bytes: bytes, ext: str = "jpg") -> str:
    os.makedirs(settings.MEDIA_DIR, exist_ok=True)
    fname = f"{uuid4().hex}.{ext}"
    fpath = os.path.join(settings.MEDIA_DIR, fname)
    with open(fpath, "wb") as f:
        f.write(file_bytes)
    # Servible vía /media/...
    return f"/media/{fname}"

@router.post("/index/url")
async def index_url(
    image_url: str = Form(...),
    price: Optional[float] = Form(None),
    location: Optional[str] = Form(""),
    store: Optional[str] = Form(""),
    tags: Optional[str] = Form("")
):
    r = requests.get(image_url, timeout=15)
    r.raise_for_status()
    img = Image.open(BytesIO(r.content)).convert("RGB")
    vec = ClipEmbedder.image_embedding(img)
    payload = {
        "image_url": image_url,
        "price": price,
        "location": location,
        "store": store,
        "tags": tags,
    }
    upsert_item(vec, payload)
    return {"status": "indexed", "image_url": image_url}

@router.post("/index/file")
async def index_file(
    file: UploadFile = File(...),
    price: Optional[float] = Form(None),
    location: Optional[str] = Form(""),
    store: Optional[str] = Form(""),
    tags: Optional[str] = Form("")
):
    content = await file.read()
    # guardar localmente y exponer por /media
    ext = (file.filename.split(".")[-1] or "jpg").lower()
    served_url = _save_local(content, ext=ext)
    img = Image.open(BytesIO(content)).convert("RGB")
    vec = ClipEmbedder.image_embedding(img)
    payload = {
        "image_url": served_url,  # URL servible desde la API
        "price": price,
        "location": location,
        "store": store,
        "tags": tags,
    }
    upsert_item(vec, payload)
    return {"status": "indexed", "image_url": served_url}
