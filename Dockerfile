FROM python:3.9-slim

WORKDIR /app

# Установка системных зависимостей для ONNX Runtime
RUN apt-get update && apt-get install -y \
    libgomp1 \
    libatlas-base-dev \
    && rm -rf /var/lib/apt/lists/*

# Копируем requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Остальной код...
