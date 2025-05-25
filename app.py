from flask import Flask, request, jsonify
import tensorflow as tf
from PIL import Image
import numpy as np
import io
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

try:
    logger.info("Memulai load model...")
    model_url = "https://storage.googleapis.com/modelta/model_cnn_mastitis.keras"
    model_path = tf.keras.utils.get_file("model_cnn_mastitis.keras", model_url)
    model = tf.keras.models.load_model(model_path)
    logger.info("Model berhasil di-load.")
except Exception as e:
    logger.error(f"Gagal memuat model: {str(e)}")
    raise

def prepare_image(img_bytes):
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        img = img.resize((240, 240))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        logger.error(f"Error memproses gambar: {str(e)}")
        raise

@app.route('/health')
def health():
    return 'OK', 200

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        image_file = request.files['image']
        img_bytes = image_file.read()
        img_array = prepare_image(img_bytes)
        prediction = model.predict(img_array)[0][0]
        label = 'mastitis' if prediction >= 0.5 else 'sehat'
        return jsonify({'prediction': label, 'score': float(prediction)})
    except Exception as e:
        logger.error(f"Error saat prediksi: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    logger.info(f"Starting Flask on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)