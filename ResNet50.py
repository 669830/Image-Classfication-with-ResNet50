!pip install datasets tensorflow

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping
from datasets import load_dataset
import numpy as np
from PIL import Image
import os

num_epochs = 10
batch_size = 64
learning_rate = 0.001
input_shape = (224, 224, 3)
num_classes = 1000

hf_dataset = load_dataset("evanarlian/imagenet_1k_resized_256")
train_ds = hf_dataset["train"].shuffle(seed=42).select(range(int(0.2 * len(hf_dataset["train"]))))
val_ds = hf_dataset["val"].shuffle(seed=42).select(range(int(0.2 * len(hf_dataset["val"]))))

def tf_generator(dataset):
    def generator():
        for example in dataset:
            image = example["image"]
            if isinstance(image, list):
                image = np.array(image)
            if not isinstance(image, Image.Image):
                image = Image.fromarray(image)

            image = image.resize((224, 224))
            image = np.array(image) / 255.0

            yield image.astype(np.float32), example["label"]
    return generator

train_tf_dataset = tf.data.Dataset.from_generator(
    tf_generator(train_ds),
    output_signature=(
        tf.TensorSpec(shape=input_shape, dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int64),
    )
).batch(batch_size).prefetch(tf.data.AUTOTUNE)

val_tf_dataset = tf.data.Dataset.from_generator(
    tf_generator(val_ds),
    output_signature=(
        tf.TensorSpec(shape=input_shape, dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int64),
    )
).batch(batch_size).prefetch(tf.data.AUTOTUNE)

# 5. Bygg ResNet50-modell med ny topp
base_model = ResNet50(include_top=False, weights='imagenet', input_shape=input_shape)
base_model.trainable = True  # Freezer feature extractor

x = layers.GlobalAveragePooling2D()(base_model.output)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(num_classes, activation='softmax')(x)

model = models.Model(inputs=base_model.input, outputs=outputs)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
history = model.fit(
    train_tf_dataset.repeat(),
    validation_data=val_tf_dataset.repeat(),
    steps_per_epoch=len(train_ds) // batch_size,
    validation_steps=len(val_ds) // batch_size,
    epochs=30,
    callbacks=[early_stop]
)
import matplotlib.pyplot as plt

# Accuracy plot
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()

# Loss plot
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()



