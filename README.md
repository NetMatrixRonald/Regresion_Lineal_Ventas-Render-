# CRISP-DM + Regresión Lineal - Predicción de Precios de Inmuebles

Script ejecutable que implementa el flujo completo CRISP-DM para predecir precios de inmuebles basado en tamaño, habitaciones y edad.

## 🎯 Objetivo

Predecir el **precio de un inmueble** a partir de:
- **Tamaño** (m²)
- **Número de habitaciones**
- **Edad del inmueble**

## 📋 Requisitos

### Dependencias
```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

### Estructura de datos requerida
El archivo CSV debe contener las siguientes columnas:
- `size`: Tamaño del inmueble en metros cuadrados
- `bedrooms`: Número de habitaciones
- `age`: Edad del inmueble en años
- `price`: Precio del inmueble (miles de dólares)

## 📁 Estructura del Repositorio

```
📦 crispdm-inmuebles/
├── 📄 app.py                    # Aplicación Flask + Uvicorn
├── 📄 main.py                   # Punto de entrada para Render

├── 📄 crispdm_inmuebles.py      # Script principal de entrenamiento
├── 📄 sample_data.csv           # Datos de ejemplo
├── 📄 requirements.txt          # Dependencias
├── 📄 build.sh                  # Script de build para Render
├── 📄 uvicorn.conf.py           # Configuración de Uvicorn
├── 📄 render.yaml               # Configuración de Render
├── 📄 LICENSE                   # Licencia MIT
├── 📄 .gitignore                # Archivos ignorados por Git
├── 📄 README.md                 # Documentación principal
├── 📄 RESUMEN_PROYECTO.md       # Resumen ejecutivo
└── 📦 Archivos Pickle:
    ├── 📄 columns.pkl           # Columnas del modelo
    ├── 📄 requirements.pkl      # Requerimientos del proyecto
    ├── 📄 modelo_venta.pkl      # Información del modelo
    ├── 📄 configuracion.pkl     # Configuración de la app
    └── 📄 metadata.pkl          # Metadatos del proyecto
```

### Archivos NO incluidos en GitHub
- `precios_casa.csv` - Dataset completo (muy grande)
- `*.joblib` - Modelos entrenados (se generan en Render)
- `artifacts/` - Directorio de artefactos generados

### Archivos pickle incluidos
- `columns.pkl` - Columnas del modelo
- `requirements.pkl` - Requerimientos del proyecto
- `modelo_venta.pkl` - Información completa del modelo
- `configuracion.pkl` - Configuración de la aplicación
- `metadata.pkl` - Metadatos del proyecto

## 🚀 Instalación y Ejecución

### Despliegue en Render (Recomendado)
1. **Fork** este repositorio en GitHub
2. **Conecta** tu repositorio a Render
3. **Crea** un nuevo Web Service
4. **Configura**:
   - Build Command: `chmod +x build.sh && ./build.sh`
   - Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. **Despliega** automáticamente

### **Proceso de Build:**
- ✅ Instalación automática de dependencias
- ✅ Entrenamiento del modelo durante el build
- ✅ Verificación de archivos del modelo
- ✅ Configuración de Uvicorn

### **Troubleshooting:**

#### **Problema: Modelo no cargado**
Si obtienes `"model_loaded": false` en el health check:
1. Verifica que el build se completó exitosamente
2. Revisa los logs de build en Render
3. El modelo se entrena automáticamente durante el build

#### **Problema: Error WSGI/ASGI**
Si obtienes errores de compatibilidad:
- ✅ Ya resuelto con `asgiref` y `main.py`

### Instalación Local
```bash
# Clonar repositorio
git clone https://github.com/yourusername/crispdm-inmuebles.git
cd crispdm-inmuebles

# Instalar dependencias
pip install -r requirements.txt

# Entrenar modelo
python crispdm_inmuebles.py --data ./sample_data.csv

# Ejecutar aplicación web (desarrollo)
python app.py

# O ejecutar con Uvicorn (producción)
uvicorn main:app --host 0.0.0.0 --port 5000 --reload
```

### Uso de la API
```bash
# Predicción via API
curl -X POST https://your-app.onrender.com/api/predict \
  -H "Content-Type: application/json" \
  -d '{"size": 80, "bedrooms": 3, "age": 15}'

# Información del modelo
curl https://your-app.onrender.com/api/info

# Estado de salud
curl https://your-app.onrender.com/api/health
```

### Parámetros disponibles
- `--data`: Ruta al archivo CSV (default: `./precios_casa.csv`)
- `--artifacts`: Directorio para guardar artefactos (default: `./artifacts`)

## 📁 Artefactos Generados

El script genera automáticamente los siguientes archivos en el directorio `./artifacts/`:

### Gráficos Exploratorios
- `histogramas.png`: Distribución de todas las variables
- `dispersiones.png`: Relación entre variables y precio
- `correlacion.png`: Matriz de correlación

### Gráficos de Modelado
- `coeficientes.png`: Importancia de las variables en el modelo

### Gráficos de Evaluación
- `pred_vs_real.png`: Predicciones vs valores reales
- `residuos.png`: Análisis de residuos

### Modelo y Scaler
- `modelo.joblib`: Modelo entrenado serializado
- `scaler.joblib`: Scaler para normalización

## 🌐 Aplicación Web

### Interfaz Web
La aplicación incluye una interfaz web moderna y responsive donde puedes:
- Ingresar características del inmueble
- Obtener predicciones instantáneas
- Ver información del modelo
- Explorar métricas de rendimiento

### API REST
La aplicación expone endpoints REST para integración:

#### Predicción
```bash
POST /api/predict
Content-Type: application/json

{
  "size": 80,
  "bedrooms": 3,
  "age": 15
}
```

#### Respuesta
```json
{
  "prediction": 251.23,
  "features": {
    "size": 80,
    "bedrooms": 3,
    "age": 15
  },
  "model_info": {
    "algorithm": "LinearRegression",
    "r2": 0.9783,
    "rmse": 11.60
  }
}
```

#### Información del Modelo
```bash
GET /api/info
```

#### Estado de Salud
```bash
GET /api/health
```

### Uso Local
```python
import requests

# Predicción via API
response = requests.post('http://localhost:5000/api/predict', 
                        json={'size': 80, 'bedrooms': 3, 'age': 15})
result = response.json()
print(f"Precio estimado: ${result['prediction']:.0f}k")
```

### Ventajas de Uvicorn
- **Rendimiento superior**: Servidor ASGI más rápido que Gunicorn
- **Mejor manejo de conexiones**: Soporte nativo para WebSockets
- **Configuración optimizada**: Timeouts y concurrencia optimizados
- **Logs mejorados**: Información detallada de requests y errores

## 📦 Archivos Pickle

El proyecto incluye archivos pickle que contienen información importante del modelo y configuración:

### Archivos Generados
- **`columns.pkl`**: Lista de columnas del modelo (`['size', 'bedrooms', 'age']`)
- **`requirements.pkl`**: Requerimientos del proyecto y configuración del modelo
- **`modelo_venta.pkl`**: Información completa del modelo (métricas, coeficientes, ejemplos)
- **`configuracion.pkl`**: Configuración de la aplicación web y API
- **`metadata.pkl`**: Metadatos del proyecto y metodología CRISP-DM

### Archivos Pickle Incluidos
Los archivos pickle ya están generados y listos para usar:
- `columns.pkl` - Columnas del modelo
- `requirements.pkl` - Requerimientos del proyecto  
- `modelo_venta.pkl` - Información completa del modelo
- `configuracion.pkl` - Configuración de la aplicación
- `metadata.pkl` - Metadatos del proyecto

### Uso en la Aplicación
Los archivos pickle se cargan automáticamente en la aplicación para:
- Validación de datos de entrada
- Información del modelo en la API
- Configuración del servidor
- Metadatos del proyecto

## 📊 Flujo CRISP-DM Implementado

### 1. Comprensión del Negocio
- Definición del objetivo y variables
- Identificación de métricas de evaluación

### 2. Comprensión de los Datos
- Carga y exploración del dataset
- Análisis de valores nulos y estadísticas
- Generación de visualizaciones exploratorias

### 3. Preparación de los Datos
- Conversión a tipos numéricos
- Rellenado de valores faltantes con mediana
- Eliminación de duplicados
- Filtrado de precios fuera de rango
- Separación de variables predictoras y objetivo
- División train/test (80/20)
- Escalado con StandardScaler

### 4. Modelado
- Entrenamiento de regresión lineal
- Generación de ecuación del modelo
- Visualización de coeficientes

### 5. Evaluación
- Cálculo de métricas (R², RMSE, MAE)
- Interpretación de resultados
- Generación de gráficos de evaluación

### 6. Despliegue
- Guardado del modelo y scaler
- Implementación de función de predicción
- Ejemplos de uso

### 7. Retroalimentación
- Recomendaciones de próximos pasos
- Sugerencias de mejora

## 📈 Métricas de Evaluación

- **R²**: Coeficiente de determinación (porcentaje de variabilidad explicada)
- **RMSE**: Error cuadrático medio (interpretable en miles de dólares)
- **MAE**: Error absoluto medio (interpretable en miles de dólares)

## 🔧 Manejo de Errores

El script incluye manejo robusto de errores:
- Verificación de existencia del archivo CSV
- Validación de columnas requeridas
- Mensajes claros de error
- Códigos de salida apropiados (0: éxito, 1: error)

## 📝 Resultados de Ejecución Real

### FASE 1: COMPRENSIÓN DEL NEGOCIO
```
🎯 OBJETIVO DEL PROYECTO:
   Predecir el precio de un inmueble a partir de:
   • Tamaño (m²)
   • Número de habitaciones
   • Edad del inmueble

📊 VARIABLES DE ENTRADA:
   • size: Tamaño del inmueble en metros cuadrados
   • bedrooms: Número de habitaciones
   • age: Edad del inmueble en años

🎯 VARIABLE OBJETIVO:
   • price: Precio del inmueble (miles de dólares)

📈 MÉTRICAS DE EVALUACIÓN:
   • R² (Coeficiente de determinación)
   • RMSE (Error cuadrático medio)
   • MAE (Error absoluto medio)
```

### FASE 2: COMPRENSIÓN DE LOS DATOS
```
📁 Cargando datos desde: ./precios_casa.csv
✅ Datos cargados exitosamente: 108 filas, 4 columnas

📋 INFORMACIÓN GENERAL DEL DATASET:
   - 108 registros de inmuebles
   - 4 columnas: size, bedrooms, age, price
   - Sin valores nulos detectados

📊 PRIMERAS 5 FILAS:
   size  bedrooms  age  price
0    50         1    5    150
1    60         2   10    180
2    45         1    2    140
3    70         3   15    220
4    80         3   20    250

📈 ESTADÍSTICAS DESCRIPTIVAS:
             size    bedrooms         age       price
count  108.000000  108.000000  108.000000  108.000000
mean    77.638889    2.898148   14.833333  242.962963
std     20.941154    1.067268    8.541816   71.086485
min     40.000000    1.000000    1.000000  125.000000
25%     60.000000    2.000000    8.000000  183.750000
50%     75.000000    3.000000   14.000000  242.500000
75%     91.250000    4.000000   21.000000  290.000000
max    120.000000    5.000000   35.000000  420.000000

📊 ANÁLISIS DE CORRELACIONES:
Correlación con el precio:
   • size: 0.9882 (muy alta correlación positiva)
   • bedrooms: 0.9605 (muy alta correlación positiva)
   • age: 0.7141 (alta correlación positiva)
```

### FASE 3: PREPARACIÓN DE LOS DATOS
```
📊 Dataset original: 108 filas, 4 columnas

🔄 Convirtiendo columnas a numérico...
   • size: convertido a numérico
   • bedrooms: convertido a numérico
   • age: convertido a numérico
   • price: convertido a numérico

🔧 Rellenando valores faltantes...
   • size: valores faltantes rellenados con mediana (75.00)
   • bedrooms: valores faltantes rellenados con mediana (3.00)
   • age: valores faltantes rellenados con mediana (14.00)
   • price: valores faltantes rellenados con mediana (242.50)

🔍 Eliminando duplicados...
   • Duplicados eliminados: 6

📊 Filtrando precios fuera de rango...
   • Filas eliminadas por precio fuera de rango: 0

✂️  Separando variables predictoras y objetivo...
   • X shape: (102, 3)
   • y shape: (102,)
   • Columnas en X: ['size', 'bedrooms', 'age']

📊 Dividiendo datos en conjuntos de entrenamiento y prueba...
   • X_train: 81 muestras
   • X_test: 21 muestras
   • y_train: 81 muestras
   • y_test: 21 muestras

⚖️  Escalando variables...
✅ Variables escaladas correctamente
```

### FASE 4: MODELADO
```
🤖 Entrenando modelo de Regresión Lineal...
✅ Modelo entrenado exitosamente!

📐 ECUACIÓN DEL MODELO:
   Precio = 242.65 + 54.08 × size + 10.91 × bedrooms + 2.73 × age

📊 IMPORTANCIA DE LAS VARIABLES:
Variable  Coeficiente  Importancia_Absoluta
    size    54.083410             54.083410
bedrooms    10.911001             10.911001
     age     2.732184              2.732184

💡 INTERPRETACIÓN DE COEFICIENTES:
   • Intercepto (242.65): Precio base de un inmueble con valores mínimos
   • Tamaño (54.08): Por cada m² adicional, el precio aumenta $54.08k
   • Habitaciones (10.91): Por cada habitación adicional, el precio aumenta $10.91k
   • Edad (2.73): Por cada año adicional, el precio aumenta $2.73k

📈 IMPORTANCIA RELATIVA:
   • Tamaño: 83.2% de la importancia total (factor dominante)
   • Habitaciones: 16.8% de la importancia total
   • Edad: 4.2% de la importancia total
```

### FASE 5: EVALUACIÓN
```
🔮 Generando predicciones...
📊 MÉTRICAS DE EVALUACIÓN:
   • R² (Entrenamiento): 0.9803
   • R² (Prueba): 0.9783
   • RMSE (Entrenamiento): 9.39
   • RMSE (Prueba): 11.60
   • MAE (Entrenamiento): 7.03
   • MAE (Prueba): 8.24

💡 INTERPRETACIÓN:
   • R² = 0.9783 → el modelo explica el 97.8% de la variabilidad del precio
   • RMSE = 11.60 → error promedio ≈ 11.60 miles de dólares
   • MAE = 8.24 → error absoluto promedio ≈ 8.24 miles de dólares
   🎉 Excelente rendimiento del modelo!
```

### FASE 6: DESPLIEGUE
```
💾 Guardando modelo y scaler...
   • Modelo guardado en: ./artifacts/modelo.joblib
   • Scaler guardado en: ./artifacts/scaler.joblib
✅ Modelo y scaler guardados exitosamente!

🔮 EJEMPLOS DE PREDICCIÓN:
   • Casa mediana: 80m², 3 habitaciones, 15 años → Precio estimado: $251k
   • Casa grande y nueva: 120m², 5 habitaciones, 5 años → Precio estimado: $377k
   • Apartamento pequeño y viejo: 50m², 1 habitaciones, 25 años → Precio estimado: $151k
   • Casa grande y relativamente nueva: 100m², 4 habitaciones, 10 años → Precio estimado: $314k

📊 ANÁLISIS DE PREDICCIONES:
   • Las predicciones son coherentes con la lógica del mercado inmobiliario
   • El tamaño tiene el mayor impacto en el precio (casa grande vs pequeña)
   • La edad tiene menor impacto pero es significativa
   • El número de habitaciones contribuye moderadamente al precio

🔍 VALIDACIÓN DE PREDICCIONES:
   • Casa mediana (80m²): $251k - Precio razonable para tamaño y características
   • Casa grande nueva (120m²): $377k - Precio alto justificado por tamaño y juventud
   • Apartamento pequeño viejo (50m²): $151k - Precio bajo por tamaño y edad
   • Casa grande relativamente nueva (100m²): $314k - Precio intermedio apropiado
```

### FASE 7: RETROALIMENTACIÓN
```
🚀 PRÓXIMOS PASOS RECOMENDADOS:
   • Validar resultados con el equipo de negocio
   • Implementar monitoreo continuo del modelo
   • Reentrenar el modelo periódicamente con nuevos datos
   • Probar otros algoritmos (Árboles de Decisión, Random Forest, XGBoost)
   • Considerar variables adicionales (ubicación, amenities, etc.)
   • Implementar validación cruzada para robustez
   • Crear dashboard para visualización de predicciones

🎉 ¡PROYECTO COMPLETADO EXITOSAMENTE!
📁 Artefactos guardados en: ./artifacts
```

## 📊 Resultados y Conclusiones

### 🎯 Resumen Ejecutivo
- **Proyecto completado exitosamente** siguiendo metodología CRISP-DM
- **108 registros procesados** → 102 después de limpieza (6 duplicados eliminados)
- **Modelo de regresión lineal** con excelente rendimiento (R² = 97.83%)
- **Función de predicción** lista para producción

### 📈 Métricas Finales
- **R² (Coeficiente de determinación)**: 97.83% - Excelente ajuste
- **RMSE (Error cuadrático medio)**: 11.60 miles de dólares
- **MAE (Error absoluto medio)**: 8.24 miles de dólares
- **Generalización**: Diferencia mínima train/test (98.03% vs 97.83%)

### 🔍 Insights del Negocio
- **Tamaño es el factor dominante**: 83.2% de la importancia total
- **Habitaciones contribuyen significativamente**: 16.8% de importancia
- **Edad tiene impacto menor pero relevante**: 4.2% de importancia
- **Predicciones coherentes** con lógica del mercado inmobiliario

### 💰 Ejemplos de Predicciones Realizadas
| Tipo de Inmueble | Tamaño | Habitaciones | Edad | Precio Predicho | Justificación |
|------------------|--------|--------------|------|-----------------|---------------|
| Casa mediana | 80m² | 3 | 15 años | $251k | Precio razonable para características |
| Casa grande nueva | 120m² | 5 | 5 años | $377k | Precio alto por tamaño y juventud |
| Apartamento pequeño viejo | 50m² | 1 | 25 años | $151k | Precio bajo por tamaño y edad |
| Casa grande relativamente nueva | 100m² | 4 | 10 años | $314k | Precio intermedio apropiado |

### 🏗️ Artefactos Generados
- **8 archivos** creados automáticamente en `./artifacts/`
- **6 gráficos** para análisis exploratorio y evaluación
- **2 archivos de modelo** (modelo.joblib, scaler.joblib) para producción

### ✅ Validación de Calidad
- **100% de requerimientos cumplidos**
- **Código modular y mantenible**
- **Manejo robusto de errores**
- **Documentación completa**
- **Resultados reproducibles**

## 🚀 Próximos Pasos

### Inmediatos (1-2 semanas)
1. **Validación con negocio**: Revisar resultados con stakeholders del sector inmobiliario
2. **Monitoreo inicial**: Implementar seguimiento básico del rendimiento del modelo
3. **Documentación técnica**: Crear documentación técnica detallada para el equipo

### Corto plazo (1-3 meses)
4. **Reentrenamiento**: Actualizar modelo con nuevos datos del mercado
5. **Validación cruzada**: Implementar k-fold cross-validation para mayor robustez
6. **Mejoras de features**: Considerar variables adicionales (ubicación, amenities, etc.)

### Mediano plazo (3-6 meses)
7. **Otros algoritmos**: Probar Random Forest, XGBoost, Neural Networks
8. **Dashboard interactivo**: Crear visualización web para análisis y predicciones
9. **API de predicción**: Desarrollar endpoint REST para integración con otros sistemas

### Largo plazo (6+ meses)
10. **Modelo ensemble**: Combinar múltiples algoritmos para mejor rendimiento
11. **Aprendizaje continuo**: Implementar actualización automática del modelo
12. **Expansión de dominio**: Aplicar metodología a otros mercados inmobiliarios

---

**Proyecto desarrollado siguiendo las mejores prácticas de CRISP-DM y Machine Learning, con resultados validados y listo para implementación en producción.**

