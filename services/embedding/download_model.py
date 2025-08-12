from transformers import CLIPProcessor, CLIPModel
import os

# Carpeta de destino (misma carpeta "model" en tu servicio)
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

print(f"Descargando modelo CLIP en {MODEL_DIR}...")

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

model.save_pretrained(MODEL_DIR)
processor.save_pretrained(MODEL_DIR)

print("Modelo descargado y guardado localmente")
