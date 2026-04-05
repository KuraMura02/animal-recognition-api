# train.py
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping
import os
import tf2onnx
import onnx

dataset_dir = "dataset"
classes = sorted(os.listdir(dataset_dir))
img_size = (224, 224)

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    horizontal_flip=True,
    rotation_range=20,
    zoom_range=0.2
)

train_generator = datagen.flow_from_directory(
    dataset_dir,
    target_size=img_size,
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    dataset_dir,
    target_size=img_size,
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# MobileNetV2 без последнего слоя
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224,224,3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(len(classes), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

early_stop = EarlyStopping(patience=3, restore_best_weights=True)
model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=20,
    callbacks=[early_stop]
)

# Создаем директории
os.makedirs("model", exist_ok=True)

# 1. Сохраняем в Keras format (на всякий случай)
model.save("model/model_mobilenet.h5")

# 2. Конвертируем в ONNX
print("Конвертация в ONNX...")

# Определяем входную сигнатуру (batch_size=None, height=224, width=224, channels=3)
spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)

# Конвертация
model_proto, _ = tf2onnx.convert.from_keras(
    model, 
    input_signature=spec,
    opset=13  # Стабильная версия ONNX opset
)

# Сохраняем ONNX модель
onnx_path = "model/model_mobilenet.onnx"
onnx.save(model_proto, onnx_path)
print(f"ONNX модель сохранена в {onnx_path}")

# 3. Проверяем ONNX модель
onnx_model = onnx.load(onnx_path)
onnx.checker.check_model(onnx_model)
print("ONNX модель валидна!")

# Сохраняем имена классов
with open("model/classes.txt", "w") as f:
    for c in classes:
        f.write(f"{c}\n")

print("Модель обучена и сконвертирована в ONNX!")
