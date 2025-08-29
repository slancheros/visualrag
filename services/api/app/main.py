from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from app.config import settings
from app.routers.indexer import router as indexer_router
from app.routers.search import router as search_router
from app.routers.agent import router as agent_router

app = FastAPI(title="Visual RAG API")

# servir imágenes indexadas localmente
app.mount("/media", StaticFiles(directory=settings.MEDIA_DIR), name="media")

# rutas
app.include_router(indexer_router, prefix="/rag", tags=["indexing"])
app.include_router(search_router, prefix="/rag", tags=["search"])
app.include_router(agent_router, tags=["agent"])

@app.get("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000)
