import streamlit as st
import h5py
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from flask import Flask, request, jsonify
from werkzeug.serving import run_simple
from threading import Thread

# Fungsi untuk memuat model K-Means dari file .h5
def load_model(filepath):
    with h5py.File(filepath, 'r') as f:
        cluster_centers = f['cluster_centers'][:]
        labels = f['labels'][:]
        inertia = f['inertia'][()]
        n_iter = f['n_iter'][()]
        n_clusters = f['n_clusters'][()]

    kmeans = KMeans(n_clusters=int(n_clusters.item()))  # Extract single element
    kmeans.cluster_centers_ = cluster_centers
    kmeans._n_threads = 1  # Set _n_threads to avoid threading issue
    kmeans.labels_ = labels
    kmeans.inertia_ = inertia
    kmeans.n_iter_ = int(n_iter.item())  # Extract single element
    return kmeans

# Muat model saat aplikasi dimulai
model = load_model('kmeans_model.h5')

# Setup Flask App within Streamlit
flask_app = Flask(__name__)

@flask_app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if 'data' not in data:
        return jsonify({"error": "Invalid input data"}), 400

    # Mengambil data dari request
    input_data = np.array(data['data'])

    if input_data.shape[1] != 50:  # Assuming 50 features
        return jsonify({"error": "Invalid input shape"}), 400

    # Melakukan prediksi
    predictions = model.predict(input_data)

    # Mengembalikan hasil prediksi dalam bentuk JSON
    return jsonify({"predictions": predictions.tolist()})

def run_flask():
    run_simple('0.0.0.0', 5000, flask_app)

# Streamlit UI
st.title("Personality Test Cluster Prediction API")
st.write("Streamlit is running alongside a Flask API. You can send POST requests to http://localhost:5000/predict")

if st.button('Start Flask API'):
    st.write("Starting Flask API...")
    thread = Thread(target=run_flask)
    thread.start()
