from fastapi import APIRouter, UploadFile, File, Form
from typing import Optional
from app.agents.orchestrator import VisualRAGAgent

router = APIRouter()

@router.post("/agent/run")
async def agent_run(
    file: Optional[UploadFile] = File(None),
    limit: int = Form(10),
    min_price: Optional[float] = Form(None),
    max_price: Optional[float] = Form(None),
    location_contains: Optional[str] = Form(None),
    store_contains: Optional[str] = Form(None),
):
    image_bytes = await file.read() if file is not None else None
    out = VisualRAGAgent.run({
        "image_bytes": image_bytes,
        "limit": limit,
        "min_price": min_price,
        "max_price": max_price,
        "location_contains": location_contains,
        "store_contains": store_contains,
    })
    return out
