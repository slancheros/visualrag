import fastapi
from fastapi import FastAPI, UploadFile, File




app = FastAPI(title="RAG Visual API", version="1.0.0")
try:
    from app.config import settings
    APP_NAME = settings.APP_NAME
except Exception:
    APP_NAME = "ragvisual-api"


@app.get("/")
async def root():
    return {"message": "Hello, World!"}

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.get("/version")
async def version():
    return {"version": "1.0.0"}





if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
