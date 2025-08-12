from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import uvicorn

import os, tempfile, shutil

from embedding import embed_image, load_model

app = FastAPI(title="Embedding Service", version="0.1.0")

@app.on_event("startup")
def _warmup():
    # Intenta cargar el modelo al iniciar (para evitar el primer request lento)
    try:
        load_model()
    except Exception as e:
        # no bloquea el arranque; se informará en /health
        app.state.model_error = str(e)

@app.get("/health")
def health():
    err = getattr(app.state, "model_error", None)
    return {"status": "ok" if err is None else "degraded", "model_error": err}

@app.post("/embed")
async def embed(file: UploadFile = File(...)):
    # Guardar a un archivo temporal
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        vec, dim = embed_image(tmp_path)
        return JSONResponse({"dim": dim, "vector": vec})
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001)