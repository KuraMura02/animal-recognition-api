#train.py
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping
import os

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

# MobileNetV2
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

os.makedirs("model", exist_ok=True)

# 1. Сохраняем в H5
model.save("model/model_mobilenet.h5")

# 2. Конвертируем в TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Оптимизация для CPU
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float32]

# Конвертируем
tflite_model = converter.convert()

# Сохраняем TFLite модель
with open("model/model_mobilenet.tflite", "wb") as f:
    f.write(tflite_model)

# Сохраняем классы
with open("model/classes.txt", "w") as f:
    for c in classes:
        f.write(f"{c}\n")

print("Модель сохранена в H5 и TFLite форматах!")
