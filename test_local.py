#!/usr/bin/env python3
"""
Script de prueba local para verificar que la aplicación funciona correctamente
"""

import requests
import json
import time

def test_local_app():
    """Prueba la aplicación localmente"""
    
    print("🧪 Iniciando pruebas locales...")
    
    # URL base local
    base_url = "http://localhost:5000"
    
    # Esperar un poco para que el servidor se inicie
    print("⏳ Esperando que el servidor se inicie...")
    time.sleep(2)
    
    try:
        # Prueba 1: Health Check
        print("\n1️⃣ Probando Health Check...")
        response = requests.get(f"{base_url}/api/health", timeout=10)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Health Check exitoso: {data}")
        else:
            print(f"   ❌ Health Check falló: {response.text}")
            return False
            
        # Prueba 2: Model Info
        print("\n2️⃣ Probando Model Info...")
        response = requests.get(f"{base_url}/api/info", timeout=10)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Model Info exitoso: {data['name']}")
        else:
            print(f"   ❌ Model Info falló: {response.text}")
            return False
            
        # Prueba 3: Predicción
        print("\n3️⃣ Probando Predicción...")
        test_data = {
            "size": 80,
            "bedrooms": 3,
            "age": 15
        }
        response = requests.post(
            f"{base_url}/api/predict",
            json=test_data,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Predicción exitosa: ${data['prediction']:.2f}")
        else:
            print(f"   ❌ Predicción falló: {response.text}")
            return False
            
        # Prueba 4: Interfaz Web
        print("\n4️⃣ Probando Interfaz Web...")
        response = requests.get(f"{base_url}/", timeout=10)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            print(f"   ✅ Interfaz Web exitosa (longitud: {len(response.text)} caracteres)")
        else:
            print(f"   ❌ Interfaz Web falló: {response.text}")
            return False
            
        print("\n🎉 ¡Todas las pruebas locales fueron exitosas!")
        return True
        
    except requests.exceptions.ConnectionError:
        print("❌ No se pudo conectar al servidor local")
        print("   Asegúrate de que el servidor esté ejecutándose con:")
        print("   python app.py")
        return False
    except Exception as e:
        print(f"❌ Error durante las pruebas: {e}")
        return False

if __name__ == "__main__":
    success = test_local_app()
    if success:
        print("\n✅ La aplicación está lista para el deploy en Render")
    else:
        print("\n❌ Hay problemas que deben resolverse antes del deploy")
