import weaviate
from app.config import settings

def get_client():
    return weaviate.Client(settings.WEAVIATE_URL)

def ensure_schema():
    client = get_client()
    cls = settings.WEAVIATE_CLASS
    schema = client.schema.get()
    if not any(c["class"] == cls for c in schema.get("classes", [])):
        client.schema.create_class({
            "class": cls,
            "description": "Visual RAG items (indexed with external vectors)",
            "vectorizer": "none",
            "properties": [
                {"name": "image_url", "dataType": ["string"]},
                {"name": "store", "dataType": ["string"]},
                {"name": "location", "dataType": ["string"]},
                {"name": "price", "dataType": ["number"]},
                {"name": "tags", "dataType": ["text"]},
            ],
        })

def upsert_item(vector, data: dict):
    client = get_client()
    return client.data_object.create(
        data_object=data,
        class_name=settings.WEAVIATE_CLASS,
        vector=vector
    )

def query_similar(vector, limit=10):
    client = get_client()
    cls = settings.WEAVIATE_CLASS
    res = (
        client.query
        .get(cls, ["image_url", "store", "location", "price", "tags"])
        .with_near_vector({"vector": vector})
        .with_additional(["distance"])
        .with_limit(limit)
        .do()
    )
    return res.get("data", {}).get("Get", {}).get(cls, [])
