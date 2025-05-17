from flask import Flask, render_template, request
import numpy as np
from PIL import Image, ImageOps

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Load model only when this route is triggered
        from tensorflow.keras.models import load_model
        model = load_model('digit_recognition/model/digit_model.h5')

        if 'file' not in request.files:
            return "No file uploaded", 400

        file = request.files['file']

        if file.filename == '':
            return "No selected file", 400

        image = Image.open(file).convert('L')  # Grayscale
        image = ImageOps.invert(image)         # Invert for MNIST-style input
        image = image.resize((28, 28))
        image = np.array(image) / 255.0
        image = image.reshape(1, 28, 28, 1)

        prediction = model.predict(image)
        digit = np.argmax(prediction)

        return render_template('result.html', digit=digit)

    except Exception as e:
        return f"Error occurred: {str(e)}", 500

if __name__ == '__main__':
    app.run(debug=True)
