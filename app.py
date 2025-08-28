#!/usr/bin/env python3
"""
Aplicación Flask para despliegue en Render
Modelo de predicción de precios de inmuebles
"""

from flask import Flask, request, jsonify, render_template_string
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path

app = Flask(__name__)

# Configuración
MODEL_PATH = './artifacts/modelo.joblib'
SCALER_PATH = './artifacts/scaler.joblib'
SAMPLE_DATA_PATH = './sample_data.csv'

# HTML template para la interfaz web
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Predicción de Precios de Inmuebles</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #2c3e50;
            text-align: center;
            margin-bottom: 30px;
        }
        .form-group {
            margin-bottom: 20px;
        }
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
            color: #34495e;
        }
        input[type="number"] {
            width: 100%;
            padding: 10px;
            border: 2px solid #ddd;
            border-radius: 5px;
            font-size: 16px;
        }
        input[type="number"]:focus {
            border-color: #3498db;
            outline: none;
        }
        button {
            background-color: #3498db;
            color: white;
            padding: 12px 30px;
            border: none;
            border-radius: 5px;
            font-size: 16px;
            cursor: pointer;
            width: 100%;
        }
        button:hover {
            background-color: #2980b9;
        }
        .result {
            margin-top: 20px;
            padding: 20px;
            background-color: #e8f5e8;
            border-radius: 5px;
            border-left: 4px solid #27ae60;
        }
        .error {
            background-color: #ffeaea;
            border-left-color: #e74c3c;
        }
        .info {
            background-color: #e3f2fd;
            border-left: 4px solid #2196f3;
            margin-bottom: 20px;
        }
        .metrics {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }
        .metric {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            text-align: center;
        }
        .metric h3 {
            margin: 0 0 10px 0;
            color: #2c3e50;
        }
        .metric p {
            margin: 0;
            font-size: 18px;
            font-weight: bold;
            color: #27ae60;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🏠 Predicción de Precios de Inmuebles</h1>
        
        <div class="info">
            <h3>📊 Información del Modelo</h3>
            <p><strong>Algoritmo:</strong> Regresión Lineal</p>
            <p><strong>R²:</strong> 97.83% | <strong>RMSE:</strong> 11.60 miles de dólares</p>
            <p><strong>Variables:</strong> Tamaño (m²), Habitaciones, Edad (años)</p>
        </div>

        <form method="POST">
            <div class="form-group">
                <label for="size">Tamaño del inmueble (m²):</label>
                <input type="number" id="size" name="size" min="40" max="120" step="1" required 
                       placeholder="Ej: 80">
            </div>
            
            <div class="form-group">
                <label for="bedrooms">Número de habitaciones:</label>
                <input type="number" id="bedrooms" name="bedrooms" min="1" max="5" step="1" required 
                       placeholder="Ej: 3">
            </div>
            
            <div class="form-group">
                <label for="age">Edad del inmueble (años):</label>
                <input type="number" id="age" name="age" min="1" max="35" step="1" required 
                       placeholder="Ej: 15">
            </div>
            
            <button type="submit">🔮 Predecir Precio</button>
        </form>

        {% if prediction %}
        <div class="result">
            <h3>💰 Resultado de la Predicción</h3>
            <p><strong>Precio estimado:</strong> ${{ "%.0f"|format(prediction) }}k</p>
            <p><strong>Características:</strong> {{ size }}m², {{ bedrooms }} habitaciones, {{ age }} años</p>
        </div>
        {% endif %}

        {% if error %}
        <div class="result error">
            <h3>❌ Error</h3>
            <p>{{ error }}</p>
        </div>
        {% endif %}

        <div class="metrics">
            <div class="metric">
                <h3>📈 R²</h3>
                <p>97.83%</p>
            </div>
            <div class="metric">
                <h3>📊 RMSE</h3>
                <p>11.60k</p>
            </div>
            <div class="metric">
                <h3>🎯 MAE</h3>
                <p>8.24k</p>
            </div>
        </div>
    </div>
</body>
</html>
"""

def load_model():
    """Cargar modelo y scaler"""
    try:
        if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
            modelo = joblib.load(MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)
            return modelo, scaler
        else:
            return None, None
    except Exception as e:
        print(f"Error cargando modelo: {e}")
        return None, None

def predict_price(size, bedrooms, age):
    """Realizar predicción de precio"""
    try:
        modelo, scaler = load_model()
        if modelo is None or scaler is None:
            return None, "Modelo no disponible. Ejecute primero el entrenamiento."
        
        # Validar rangos
        if not (40 <= size <= 120):
            return None, "Tamaño debe estar entre 40 y 120 m²"
        if not (1 <= bedrooms <= 5):
            return None, "Habitaciones debe estar entre 1 y 5"
        if not (1 <= age <= 35):
            return None, "Edad debe estar entre 1 y 35 años"
        
        # Preparar datos
        datos = np.array([[size, bedrooms, age]])
        datos_escalados = scaler.transform(datos)
        
        # Predicción
        precio_predicho = modelo.predict(datos_escalados)[0]
        
        return precio_predicho, None
    except Exception as e:
        return None, f"Error en predicción: {str(e)}"

@app.route('/')
def home():
    """Página principal"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/', methods=['POST'])
def predict():
    """Endpoint para predicción"""
    try:
        # Obtener datos del formulario
        size = float(request.form['size'])
        bedrooms = int(request.form['bedrooms'])
        age = int(request.form['age'])
        
        # Realizar predicción
        prediction, error = predict_price(size, bedrooms, age)
        
        if error:
            return render_template_string(HTML_TEMPLATE, error=error)
        
        return render_template_string(
            HTML_TEMPLATE, 
            prediction=prediction,
            size=size,
            bedrooms=bedrooms,
            age=age
        )
        
    except Exception as e:
        return render_template_string(HTML_TEMPLATE, error=f"Error: {str(e)}")

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint para predicción"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No se proporcionaron datos'}), 400
        
        size = float(data.get('size', 0))
        bedrooms = int(data.get('bedrooms', 0))
        age = int(data.get('age', 0))
        
        prediction, error = predict_price(size, bedrooms, age)
        
        if error:
            return jsonify({'error': error}), 400
        
        return jsonify({
            'prediction': round(prediction, 2),
            'features': {
                'size': size,
                'bedrooms': bedrooms,
                'age': age
            },
            'model_info': {
                'algorithm': 'LinearRegression',
                'r2': 0.9783,
                'rmse': 11.60
            }
        })
        
    except Exception as e:
        return jsonify({'error': f'Error interno: {str(e)}'}), 500

@app.route('/api/health')
def health():
    """Endpoint de salud"""
    modelo, scaler = load_model()
    status = "healthy" if modelo is not None and scaler is not None else "unhealthy"
    
    return jsonify({
        'status': status,
        'model_loaded': modelo is not None,
        'scaler_loaded': scaler is not None
    })

@app.route('/api/info')
def info():
    """Información del modelo"""
    return jsonify({
        'name': 'Modelo de Predicción de Precios de Inmuebles',
        'version': '1.0.0',
        'algorithm': 'LinearRegression',
        'features': ['size', 'bedrooms', 'age'],
        'target': 'price',
        'metrics': {
            'r2': 0.9783,
            'rmse': 11.60,
            'mae': 8.24
        },
        'equation': 'Precio = 242.65 + 54.08 × size + 10.91 × bedrooms + 2.73 × age'
    })

if __name__ == '__main__':
    # Crear directorio de artefactos si no existe
    Path('./artifacts').mkdir(exist_ok=True)
    
    # Verificar si el modelo existe
    modelo, scaler = load_model()
    if modelo is None:
        print("⚠️  Modelo no encontrado. Ejecute primero el entrenamiento.")
        print("   python crispdm_inmuebles.py --data ./sample_data.csv")
    
    # Configurar puerto para Render
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
