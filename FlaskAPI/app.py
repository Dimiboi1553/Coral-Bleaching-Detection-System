from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
import os
import uuid

app = Flask(__name__)

model = tf.keras.models.load_model('FinishedModel.h5')

@app.route('/')
def index():
    return render_template('main.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the uploaded image file
        file = request.files['file']

        # Generate a unique filename for each uploaded image
        unique_filename = str(uuid.uuid4()) + '.jpg'
        image_path = os.path.join('static', unique_filename)

        file.save(image_path)

        # Preprocess the image using OpenCV
        img = cv2.imread(image_path)
        img = cv2.resize(img, (300, 300))
        img = img / 255.0  # Normalize pixel values

        # Expand dimensions to match model input shape
        img_array = np.expand_dims(img, axis=0)

        # Make predictions using the loaded model
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)[0]

        # Map numeric class to human-readable label
        result = "Healthy" if predicted_class == 0 else "Bleached"

        # Return the prediction as JSON
        return render_template('results.html', prediction=result, image_path=unique_filename)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
