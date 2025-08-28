#!/usr/bin/env bash
# Script de build para Render

echo "🚀 Iniciando build para Render..."

# Instalar dependencias
echo "📦 Instalando dependencias..."
pip install -r requirements.txt

# Crear directorio de artefactos
echo "📁 Creando directorio de artefactos..."
mkdir -p artifacts

# Entrenar modelo
echo "🤖 Entrenando modelo..."
python train_model.py

# Verificar que el modelo se creó correctamente
if [ -f "artifacts/modelo.joblib" ] && [ -f "artifacts/scaler.joblib" ]; then
    echo "✅ Modelo entrenado exitosamente"
    echo "📊 Archivos generados:"
    ls -la artifacts/
else
    echo "❌ Error: No se pudo entrenar el modelo"
    echo "⚠️  Continuando sin modelo entrenado..."
fi

# Verificar instalación de Uvicorn
echo "🔧 Verificando Uvicorn..."
python -c "import uvicorn; print(f'✅ Uvicorn {uvicorn.__version__} instalado')"

echo "🎉 Build completado exitosamente!"
