#!/usr/bin/env python3
"""
Script de entrenamiento manual para Render
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

def train_model():
    """Entrenar modelo manualmente"""
    
    print("🤖 Entrenando modelo manualmente...")
    
    # Crear datos de ejemplo
    data = {
        'size': [50, 60, 45, 70, 80, 90, 55, 65, 75, 85],
        'bedrooms': [1, 2, 1, 3, 3, 4, 2, 2, 3, 4],
        'age': [5, 10, 2, 15, 20, 25, 8, 12, 18, 22],
        'price': [150, 180, 140, 220, 250, 300, 160, 190, 240, 280]
    }
    df = pd.DataFrame(data)
    
    print(f"📊 Datos creados: {len(df)} filas")
    
    # Preparar datos
    X = df[['size', 'bedrooms', 'age']]
    y = df['price']
    
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Escalar
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Entrenar modelo
    modelo = LinearRegression()
    modelo.fit(X_train_scaled, y_train)
    
    # Evaluar modelo
    X_test_scaled = scaler.transform(X_test)
    y_pred = modelo.predict(X_test_scaled)
    
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"📈 Métricas del modelo:")
    print(f"   R²: {r2:.4f}")
    print(f"   RMSE: {rmse:.2f}")
    print(f"   MAE: {mae:.2f}")
    
    # Crear directorio artifacts si no existe
    os.makedirs('artifacts', exist_ok=True)
    
    # Guardar modelo y scaler
    joblib.dump(modelo, 'artifacts/modelo.joblib')
    joblib.dump(scaler, 'artifacts/scaler.joblib')
    
    print("✅ Modelo entrenado y guardado exitosamente")
    print(f"📁 Archivos guardados en: {os.path.abspath('artifacts')}")
    
    # Verificar que los archivos existen
    if os.path.exists('artifacts/modelo.joblib') and os.path.exists('artifacts/scaler.joblib'):
        print("✅ Verificación: Archivos del modelo creados correctamente")
        return True
    else:
        print("❌ Error: No se pudieron crear los archivos del modelo")
        return False

if __name__ == "__main__":
    success = train_model()
    if success:
        print("🎉 Entrenamiento completado exitosamente")
    else:
        print("❌ Error en el entrenamiento")
        exit(1)
