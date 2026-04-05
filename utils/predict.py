# utils/predict.py
import tensorflow as tf
import numpy as np
from PIL import Image
import io

model = None
classes = None

def load_model_once():
    global model, classes
    if model is None:
        # compile=False убирает проблемы с параметрами BatchNormalization
        model = tf.keras.models.load_model("model/model_mobilenet.h5", compile=False)
        with open("classes.txt", "r", encoding="utf-8") as f:
            classes = [line.strip() for line in f]

def predict_animal(image_bytes):
    load_model_once()

    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((224, 224))
    x = np.array(img, dtype=np.float32) / 255.0
    x = np.expand_dims(x, axis=0)

    pred = model.predict(x)
    class_idx = np.argmax(pred)

    return {
        "animal": classes[class_idx],
        "confidence": float(pred[0][class_idx])
    }
