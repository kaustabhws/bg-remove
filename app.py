import os
from flask import Flask, request, send_file
import cv2
import numpy as np
import tensorflow_hub as hub
import tensorflow as tf
from io import BytesIO

app = Flask(__name__)

# Load the model
model = hub.KerasLayer("https://www.kaggle.com/models/vaishaknair456/u2-net-portrait-background-remover/tensorFlow2/40_saved_model/1")

# Constants
INPUT_IMG_HEIGHT = 512
INPUT_IMG_WIDTH = 512
INPUT_CHANNEL_COUNT = 3
PROBABILITY_THRESHOLD = 0.7

@app.route('/remove-background', methods=['POST'])
def remove_background():
    if 'file' not in request.files:
        return "No file provided", 400

    file = request.files['file']
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    h, w, channel_count = image.shape

    # Preprocess input image
    if channel_count > INPUT_CHANNEL_COUNT:
        image = image[..., :INPUT_CHANNEL_COUNT]

    x = cv2.resize(image, (INPUT_IMG_WIDTH, INPUT_IMG_HEIGHT))
    x = x / 255.0
    x = x.astype(np.float32)
    x = np.expand_dims(x, axis=0)

    # Generate the pixel-wise probability mask
    probability = model(x)[0].numpy()

    # Produce output image
    probability = cv2.resize(probability, dsize=(w, h))
    probability = np.expand_dims(probability, axis=-1)

    alpha_image = np.insert(image, 3, 255.0, axis=2)
    masked_image = np.where(probability > PROBABILITY_THRESHOLD, alpha_image, 0.0).astype(np.uint8)

    # Save output to a BytesIO object
    success, buffer = cv2.imencode('.png', masked_image)
    if not success:
        return "Failed to encode image", 500
    output_buffer = BytesIO(buffer)

    return send_file(output_buffer, mimetype='image/png')

