#!/usr/bin/env bash
# Script de build para Render

echo "🚀 Iniciando build para Render..."

# Instalar dependencias
echo "📦 Instalando dependencias..."
pip install -r requirements.txt

# Verificar instalación de Uvicorn
echo "🔧 Verificando Uvicorn..."
python -c "import uvicorn; print(f'✅ Uvicorn {uvicorn.__version__} instalado')"

echo "🎉 Build completado exitosamente!"
