import tensorflow as tf
import os

BASE_DIR = os.getcwd()
MODEL_DIR = os.path.join(BASE_DIR, "Model")
KERAS_MODEL_PATH = os.path.join(MODEL_DIR, "keras_model.h5")
TFLITE_OUT_PATH = os.path.join(BASE_DIR, "model.tflite")

model = tf.keras.models.load_model(KERAS_MODEL_PATH)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open(TFLITE_OUT_PATH, "wb") as f:
    f.write(tflite_model)