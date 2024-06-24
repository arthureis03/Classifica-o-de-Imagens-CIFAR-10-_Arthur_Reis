from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io

# Carregamento o modelo treinado
model = load_model('cifar10_model.h5')

# Definição as classes
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

app = Flask(__name__)

def preprocess_image(image):
    image = image.resize((32, 32))
    image = np.array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    image = Image.open(io.BytesIO(file.read()))
    processed_image = preprocess_image(image)
    
    predictions = model.predict(processed_image)
    predicted_label = np.argmax(predictions)
    confidence = np.max(predictions)
    
    response = {
        "category": class_names[predicted_label],
        "confidence": float(confidence)
    }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
