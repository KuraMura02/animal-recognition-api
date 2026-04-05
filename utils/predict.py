import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

# Загружаем TFLite модель
model_path = os.path.join(os.path.dirname(__file__), "..", "model", "model_mobilenet.tflite")
classes_path = os.path.join(os.path.dirname(__file__), "..", "model", "classes.txt")

# Загружаем интерпретатор
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Получаем детали входного и выходного слоев
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Загружаем классы
with open(classes_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]

def predict_animal(image_bytes):
    """
    Предсказание животного по изображению используя TFLite
    """
    # Загружаем и预处理 изображение
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((224, 224))
    
    # Нормализация
    x = np.array(img, dtype=np.float32) / 255.0
    x = np.expand_dims(x, axis=0)
    
    # Устанавливаем входные данные
    interpreter.set_tensor(input_details[0]['index'], x)
    
    # Выполняем инференс
    interpreter.invoke()
    
    # Получаем результат
    pred = interpreter.get_tensor(output_details[0]['index'])
    
    class_idx = np.argmax(pred[0])
    confidence = float(pred[0][class_idx])
    
    return {
        "animal": classes[class_idx],
        "confidence": confidence
    }
