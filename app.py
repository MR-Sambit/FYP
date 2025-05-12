from flask import Flask, request, render_template, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)

# Load class labels from labels.txt
with open("labels.txt", "r") as f:
    LABELS = [line.strip() for line in f.readlines()]

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def preprocess_image(image, target_size):
    image = image.convert("RGB")
    image = image.resize(target_size)
    img_array = np.array(image, dtype=np.float32)
    if input_details[0]['dtype'] == np.float32:
        img_array /= 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    image = Image.open(file.stream)
    input_shape = input_details[0]['shape'][1:3]
    processed_image = preprocess_image(image, input_shape)

    interpreter.set_tensor(input_details[0]['index'], processed_image)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    prediction = np.squeeze(output_data)
    predicted_index = int(np.argmax(prediction))
    predicted_label = LABELS[predicted_index] if predicted_index < len(LABELS) else f"Class {predicted_index}"
    confidence = float(np.max(prediction))

    return jsonify({
        "class_id": predicted_index,
        "class_name": predicted_label,
        "confidence": round(confidence * 100, 2)
    })

if __name__ == '__main__':
    app.run(debug=True)
