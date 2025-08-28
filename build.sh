#!/usr/bin/env bash
# Script de build para Render

echo "🚀 Iniciando build para Render..."

# Instalar dependencias
echo "📦 Instalando dependencias..."
pip install -r requirements.txt

# Crear directorio de artefactos
echo "📁 Creando directorio de artefactos..."
mkdir -p artifacts

# Entrenar modelo si no existe
if [ ! -f "artifacts/modelo.joblib" ] || [ ! -f "artifacts/scaler.joblib" ]; then
    echo "🤖 Entrenando modelo..."
    python crispdm_inmuebles.py --data ./sample_data.csv
else
    echo "✅ Modelo ya existe, saltando entrenamiento..."
fi

# Verificar que el modelo se creó correctamente
if [ -f "artifacts/modelo.joblib" ] && [ -f "artifacts/scaler.joblib" ]; then
    echo "✅ Modelo entrenado exitosamente"
    echo "📊 Archivos generados:"
    ls -la artifacts/
else
    echo "❌ Error: No se pudo entrenar el modelo"
    exit 1
fi

# Verificar instalación de Uvicorn
echo "🔧 Verificando Uvicorn..."
python -c "import uvicorn; print(f'✅ Uvicorn {uvicorn.__version__} instalado')"

# Crear directorio de artefactos
echo "📁 Creando directorio de artefactos..."
mkdir -p artifacts

# Entrenar modelo
echo "🤖 Entrenando modelo..."
python crispdm_inmuebles.py --data ./sample_data.csv

# Verificar que el modelo se creó correctamente
if [ -f "artifacts/modelo.joblib" ] && [ -f "artifacts/scaler.joblib" ]; then
    echo "✅ Modelo entrenado exitosamente"
    echo "📊 Archivos generados:"
    ls -la artifacts/
else
    echo "❌ Error: No se pudo entrenar el modelo"
    exit 1
fi

echo "🎉 Build completado exitosamente!"
