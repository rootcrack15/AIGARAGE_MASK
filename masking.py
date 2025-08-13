import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

# Model yükle
interpreter = tf.lite.Interpreter(model_path="deeplabv3-xception65.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def run_mask(image_path):
    img = Image.open(image_path).convert("RGB").resize((224, 224))
    input_data = np.expand_dims(img, axis=0).astype(np.float32) / 255.0

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])[0]
    mask = (output_data > 0.5).astype(np.uint8) * 255

    cv2.imwrite("mask.png", mask)
    print("Maske oluşturuldu: mask.png")

if __name__ == "__main__":
    run_mask("test.jpg")
