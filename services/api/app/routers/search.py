from fastapi import APIRouter, UploadFile, File, Form
from typing import Optional
from io import BytesIO
from PIL import Image
from app.services.clip_embedder import ClipEmbedder
from app.services.weaviate_client import query_similar

router = APIRouter()

@router.post("/search/image")
async def search_image(
    file: UploadFile = File(...),
    limit: int = Form(10),
    min_price: Optional[float] = Form(None),
    max_price: Optional[float] = Form(None),
    location_contains: Optional[str] = Form(None),
    store_contains: Optional[str] = Form(None)
):
    content = await file.read()
    img = Image.open(BytesIO(content)).convert("RGB")
    vec = ClipEmbedder.image_embedding(img)
    hits = query_similar(vec, limit=limit)

    # Post-filtros opcionales en la API
    def ok(h):
        p = h.get("price")
        loc = (h.get("location") or "").lower()
        st = (h.get("store") or "").lower()
        cond = True
        if min_price is not None:
            cond &= (p is not None and p >= min_price)
        if max_price is not None:
            cond &= (p is not None and p <= max_price)
        if location_contains:
            cond &= (location_contains.lower() in loc)
        if store_contains:
            cond &= (store_contains.lower() in st)
        return cond

    filtered = [h for h in hits if ok(h)]
    # convertir distancia (cosine distance) a similitud aproximada
    results = []
    for h in filtered:
        add = h.get("_additional", {}) or {}
        distance = add.get("distance", None)
        similarity = None
        if distance is not None:
            try:
                similarity = float(1.0 - float(distance))
            except:
                similarity = None
        out = {
            "image_url": h.get("image_url"),
            "store": h.get("store"),
            "location": h.get("location"),
            "price": h.get("price"),
            "tags": h.get("tags"),
            "distance": distance,
            "similarity": similarity,
        }
        results.append(out)

    return {"count": len(results), "results": results[:limit]}
