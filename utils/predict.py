# utils/predict.py
import tensorflow as tf
import numpy as np
from PIL import Image
import io

model = tf.keras.models.load_model("model/model_mobilenet.h5")
classes = [line.strip() for line in open("classes.txt")]

def predict_animal(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((224,224))
    x = np.array(img)/255.0
    x = np.expand_dims(x, axis=0)

    pred = model.predict(x)
    class_idx = np.argmax(pred)
    return {
        "animal": classes[class_idx],
        "confidence": float(pred[0][class_idx])
    }
