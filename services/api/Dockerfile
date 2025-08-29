FROM python:3.11-slim

WORKDIR /app

# deps del sistema para torch/vision e imágenes
RUN apt-get update && apt-get install -y \
    gcc git libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /app/media
COPY app ./app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
EXPOSE 8000

