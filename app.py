from flask import Flask, request, jsonify
from prometheus_client import Counter, generate_latest, make_wsgi_app
from werkzeug.middleware.dispatcher import DispatcherMiddleware
import pickle
import numpy as np
import os

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# Path model
MODEL_PATH = "/app/model_serving/1732608801/model.pkl"  # Path di Docker/Heroku

# Fungsi untuk memuat model
def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    with open(model_path, 'rb') as f:
        return pickle.load(f)

# Memuat model saat aplikasi Flask berjalan
model = load_model(MODEL_PATH)

# Prometheus metrics
REQUEST_COUNT = Counter(
    'request_count', 
    'Total HTTP Requests',
    ['method', 'endpoint', 'http_status']
)

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint untuk melakukan prediksi."""
    try:
        input_data = request.json.get('data')
        if not input_data:
            REQUEST_COUNT.labels(method='POST', endpoint='/predict', http_status=400).inc()
            return jsonify({'error': 'No data provided'}), 400

        # Konversi data ke format numpy
        input_array = np.array(input_data).reshape(1, -1)

        # Melakukan prediksi
        prediction = model.predict(input_array)

        # Update Prometheus metrics
        REQUEST_COUNT.labels(method='POST', endpoint='/predict', http_status=200).inc()
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        REQUEST_COUNT.labels(method='POST', endpoint='/predict', http_status=500).inc()
        return jsonify({'error': str(e)}), 500

@app.route('/metrics', methods=['GET'])
def metrics():
    """Endpoint untuk monitoring Prometheus."""
    return generate_latest(), 200

# Integrasi Prometheus middleware
app.wsgi_app = DispatcherMiddleware(app.wsgi_app, {
    '/metrics': make_wsgi_app()
})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
