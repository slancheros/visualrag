from typing import Dict, Any
from io import BytesIO
from PIL import Image
from app.services.clip_embedder import ClipEmbedder
from app.services.weaviate_client import query_similar

class VisualRAGAgent:
    """
    Orquestador simple basado en reglas.
    inputs esperados:
      - image_bytes: bytes de imagen (opcional)
      - limit, min_price, max_price, location_contains, store_contains (opcionales)
    """
    @staticmethod
    def run(inputs: Dict[str, Any]) -> Dict[str, Any]:
        image_bytes = inputs.get("image_bytes")
        limit = int(inputs.get("limit", 10))
        min_price = inputs.get("min_price")
        max_price = inputs.get("max_price")
        location_contains = inputs.get("location_contains")
        store_contains = inputs.get("store_contains")

        if image_bytes:
            # tool: search_similar
            img = Image.open(BytesIO(image_bytes)).convert("RGB")
            vec = ClipEmbedder.image_embedding(img)
            hits = query_similar(vec, limit=limit)

            def ok(h):
                p = h.get("price")
                loc = (h.get("location") or "").lower()
                st  = (h.get("store") or "").lower()
                cond = True
                if min_price is not None:
                    cond &= (p is not None and p >= float(min_price))
                if max_price is not None:
                    cond &= (p is not None and p <= float(max_price))
                if location_contains:
                    cond &= (str(location_contains).lower() in loc)
                if store_contains:
                    cond &= (str(store_contains).lower() in st)
                return cond

            filtered = [h for h in hits if ok(h)]
            results = []
            for h in filtered:
                d = (h.get("_additional") or {}).get("distance")
                sim = None
                if d is not None:
                    try: sim = 1.0 - float(d)
                    except: pass
                results.append({
                    "image_url": h.get("image_url"),
                    "store": h.get("store"),
                    "location": h.get("location"),
                    "price": h.get("price"),
                    "tags": h.get("tags"),
                    "distance": d,
                    "similarity": sim,
                })
            return {"tool_used": "search_similar", "count": len(results), "results": results[:limit]}

        # (futuro) aquí decidiríamos usar otras tools (OCR, scraping, catálogos, etc.)
        return {"tool_used": None, "message": "No hay imagen. En próximos pasos añadimos otras herramientas."}
