import os
from pydantic import BaseModel

class Settings(BaseModel):
    WEAVIATE_URL: str = os.getenv("WEAVIATE_URL", "http://localhost:8080")
    WEAVIATE_CLASS: str = os.getenv("WEAVIATE_CLASS", "CatalogItem")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "openai/clip-vit-base-patch32")
    MEDIA_DIR: str = os.getenv("MEDIA_DIR", "/app/media")

settings = Settings()
print(f"Configuration loaded: {settings}")