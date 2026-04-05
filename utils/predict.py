# utils/predict.py
import onnxruntime as ort
import numpy as np
from PIL import Image
import io
import os

# Загружаем ONNX модель (более стабильно, чем TensorFlow)
model_path = os.path.join(os.path.dirname(__file__), "..", "model", "model_mobilenet.onnx")
classes_path = os.path.join(os.path.dirname(__file__), "..", "model", "classes.txt")

# Создаем сессию ONNX Runtime
session = ort.InferenceSession(model_path)

# Загружаем классы
with open(classes_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Получаем имя входного слоя
input_name = session.get_inputs()[0].name


def predict_animal(image_bytes):
    """
    Предсказание животного по изображению
    """
    # Загружаем и预处理 изображение
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((224, 224))

    # Нормализация и преобразование в numpy
    x = np.array(img, dtype=np.float32) / 255.0

    # Добавляем batch dimension
    x = np.expand_dims(x, axis=0)

    # Инференс через ONNX Runtime
    pred = session.run(None, {input_name: x})[0]

    # Получаем предсказанный класс
    class_idx = np.argmax(pred[0])
    confidence = float(pred[0][class_idx])

    return {
        "animal": classes[class_idx],
        "confidence": confidence
    }
