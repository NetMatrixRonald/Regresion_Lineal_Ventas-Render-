#!/usr/bin/env python3
"""
CRISP-DM + Regresión Lineal - Predicción de Precios de Inmuebles
Script ejecutable que implementa el flujo completo CRISP-DM para predecir precios
basado en tamaño, habitaciones y edad del inmueble.

Autor: Senior Data Scientist
Fecha: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import joblib
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuración de matplotlib
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10

def fase1_comprension_negocio():
    """
    FASE 1: COMPRENSIÓN DEL NEGOCIO
    """
    print("="*80)
    print("FASE 1: COMPRENSIÓN DEL NEGOCIO")
    print("="*80)
    print("🎯 OBJETIVO DEL PROYECTO:")
    print("   Predecir el precio de un inmueble a partir de:")
    print("   • Tamaño (m²)")
    print("   • Número de habitaciones")
    print("   • Edad del inmueble")
    print()
    print("📊 VARIABLES DE ENTRADA:")
    print("   • size: Tamaño del inmueble en metros cuadrados")
    print("   • bedrooms: Número de habitaciones")
    print("   • age: Edad del inmueble en años")
    print()
    print("🎯 VARIABLE OBJETIVO:")
    print("   • price: Precio del inmueble (miles de dólares)")
    print()
    print("📈 MÉTRICAS DE EVALUACIÓN:")
    print("   • R² (Coeficiente de determinación)")
    print("   • RMSE (Error cuadrático medio)")
    print("   • MAE (Error absoluto medio)")
    print()

def cargar_datos(path):
    """
    FASE 2: COMPRENSIÓN DE LOS DATOS - Carga de datos
    """
    print("="*80)
    print("FASE 2: COMPRENSIÓN DE LOS DATOS")
    print("="*80)
    
    try:
        print(f"📁 Cargando datos desde: {path}")
        df = pd.read_csv(path)
        print(f"✅ Datos cargados exitosamente: {df.shape[0]} filas, {df.shape[1]} columnas")
        return df
    except FileNotFoundError:
        print(f"❌ Error: No se encontró el archivo {path}")
        return None
    except Exception as e:
        print(f"❌ Error al cargar datos: {e}")
        return None

def explorar_datos(df):
    """
    FASE 2: COMPRENSIÓN DE LOS DATOS - Exploración
    """
    print("\n📋 INFORMACIÓN GENERAL DEL DATASET:")
    print(df.info())
    
    print("\n📊 PRIMERAS 5 FILAS:")
    print(df.head())
    
    print("\n📈 ESTADÍSTICAS DESCRIPTIVAS:")
    print(df.describe())
    
    print("\n🔍 ANÁLISIS DE VALORES NULOS:")
    valores_nulos = df.isnull().sum()
    print(valores_nulos)
    
    if valores_nulos.sum() > 0:
        print("⚠️  Se encontraron valores nulos que serán manejados en la preparación")
    else:
        print("✅ No se encontraron valores nulos")
    
    # Verificar columnas requeridas
    columnas_requeridas = ["size", "bedrooms", "age", "price"]
    columnas_faltantes = [col for col in columnas_requeridas if col not in df.columns]
    
    if columnas_faltantes:
        print(f"\n❌ ERROR: Faltan columnas requeridas: {columnas_faltantes}")
        return False
    
    print(f"\n✅ Todas las columnas requeridas están presentes")
    return True

def generar_graficos_exploratorios(df, artifacts_dir):
    """
    Generar y guardar gráficos exploratorios
    """
    print("\n📈 Generando gráficos exploratorios...")
    
    # Crear directorio si no existe
    Path(artifacts_dir).mkdir(parents=True, exist_ok=True)
    
    # Histogramas
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Distribución de Variables', fontsize=16, fontweight='bold')
    
    axes[0, 0].hist(df['size'], bins=15, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Distribución del Tamaño')
    axes[0, 0].set_xlabel('Tamaño (m²)')
    axes[0, 0].set_ylabel('Frecuencia')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].hist(df['bedrooms'], bins=range(1, 7), alpha=0.7, color='lightgreen', edgecolor='black')
    axes[0, 1].set_title('Distribución de Habitaciones')
    axes[0, 1].set_xlabel('Número de Habitaciones')
    axes[0, 1].set_ylabel('Frecuencia')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].hist(df['age'], bins=15, alpha=0.7, color='salmon', edgecolor='black')
    axes[1, 0].set_title('Distribución de la Edad')
    axes[1, 0].set_xlabel('Edad (años)')
    axes[1, 0].set_ylabel('Frecuencia')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].hist(df['price'], bins=15, alpha=0.7, color='orange', edgecolor='black')
    axes[1, 1].set_title('Distribución del Precio')
    axes[1, 1].set_xlabel('Precio (miles $)')
    axes[1, 1].set_ylabel('Frecuencia')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{artifacts_dir}/histogramas.png', dpi=300, bbox_inches='tight')
    print("✅ Histogramas guardados como 'histogramas.png'")
    plt.close()
    
    # Gráficos de dispersión
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Relación con el Precio', fontsize=16, fontweight='bold')
    
    axes[0].scatter(df['size'], df['price'], alpha=0.6, color='blue')
    axes[0].set_title('Tamaño vs Precio')
    axes[0].set_xlabel('Tamaño (m²)')
    axes[0].set_ylabel('Precio (miles $)')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].scatter(df['bedrooms'], df['price'], alpha=0.6, color='green')
    axes[1].set_title('Habitaciones vs Precio')
    axes[1].set_xlabel('Número de Habitaciones')
    axes[1].set_ylabel('Precio (miles $)')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].scatter(df['age'], df['price'], alpha=0.6, color='red')
    axes[2].set_title('Edad vs Precio')
    axes[2].set_xlabel('Edad (años)')
    axes[2].set_ylabel('Precio (miles $)')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{artifacts_dir}/dispersiones.png', dpi=300, bbox_inches='tight')
    print("✅ Gráficos de dispersión guardados como 'dispersiones.png'")
    plt.close()
    
    # Matriz de correlación
    correlacion = df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlacion, annot=True, cmap='coolwarm', center=0, 
               square=True, linewidths=0.5, fmt='.3f')
    plt.title('Matriz de Correlación', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{artifacts_dir}/correlacion.png', dpi=300, bbox_inches='tight')
    print("✅ Matriz de correlación guardada como 'correlacion.png'")
    plt.close()
    
    # Análisis de correlaciones
    print("\n📊 ANÁLISIS DE CORRELACIONES:")
    print("Correlación con el precio:")
    for col in ['size', 'bedrooms', 'age']:
        corr = df[col].corr(df['price'])
        print(f"   • {col}: {corr:.4f}")

def preparar_datos(df):
    """
    FASE 3: PREPARACIÓN DE LOS DATOS
    """
    print("\n" + "="*80)
    print("FASE 3: PREPARACIÓN DE LOS DATOS")
    print("="*80)
    
    df_limpio = df.copy()
    print(f"📊 Dataset original: {df_limpio.shape[0]} filas, {df_limpio.shape[1]} columnas")
    
    # 1. Convertir a numérico con coerce
    print("\n🔄 Convirtiendo columnas a numérico...")
    columnas_numericas = ["size", "bedrooms", "age", "price"]
    for col in columnas_numericas:
        if col in df_limpio.columns:
            df_limpio[col] = pd.to_numeric(df_limpio[col], errors="coerce")
            print(f"   • {col}: convertido a numérico")
    
    # 2. Rellenar valores faltantes con mediana
    print("\n🔧 Rellenando valores faltantes...")
    for col in columnas_numericas:
        if col in df_limpio.columns:
            mediana = df_limpio[col].median()
            df_limpio[col].fillna(mediana, inplace=True)
            print(f"   • {col}: valores faltantes rellenados con mediana ({mediana:.2f})")
    
    # 3. Eliminar duplicados
    print("\n🔍 Eliminando duplicados...")
    filas_antes = len(df_limpio)
    df_limpio.drop_duplicates(inplace=True)
    filas_despues = len(df_limpio)
    print(f"   • Duplicados eliminados: {filas_antes - filas_despues}")
    
    # 4. Filtrar precios fuera de rango
    print("\n📊 Filtrando precios fuera de rango...")
    filas_antes = len(df_limpio)
    df_limpio = df_limpio[(df_limpio["price"] > 0) & (df_limpio["price"] < 1000)]
    filas_despues = len(df_limpio)
    print(f"   • Filas eliminadas por precio fuera de rango: {filas_antes - filas_despues}")
    
    # 5. Separar X e y
    print("\n✂️  Separando variables predictoras y objetivo...")
    X = df_limpio[["size", "bedrooms", "age"]]
    y = df_limpio["price"]
    
    print(f"   • X shape: {X.shape}")
    print(f"   • y shape: {y.shape}")
    print(f"   • Columnas en X: {list(X.columns)}")
    
    return X, y

def dividir_datos(X, y):
    """
    Dividir datos en train y test
    """
    print("\n📊 Dividiendo datos en conjuntos de entrenamiento y prueba...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"   • X_train: {X_train.shape[0]} muestras")
    print(f"   • X_test: {X_test.shape[0]} muestras")
    print(f"   • y_train: {y_train.shape[0]} muestras")
    print(f"   • y_test: {y_test.shape[0]} muestras")
    
    return X_train, X_test, y_train, y_test

def escalar_datos(X_train, X_test):
    """
    Escalar variables con StandardScaler
    """
    print("\n⚖️  Escalando variables...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("✅ Variables escaladas correctamente")
    return X_train_scaled, X_test_scaled, scaler

def entrenar_modelo(X_train, y_train):
    """
    FASE 4: MODELADO
    """
    print("\n" + "="*80)
    print("FASE 4: MODELADO")
    print("="*80)
    
    print("🤖 Entrenando modelo de Regresión Lineal...")
    modelo = LinearRegression()
    modelo.fit(X_train, y_train)
    print("✅ Modelo entrenado exitosamente!")
    
    # Obtener coeficientes
    coeficientes = modelo.coef_
    intercepto = modelo.intercept_
    
    print(f"\n📐 ECUACIÓN DEL MODELO:")
    ecuacion = f"Precio = {intercepto:.2f}"
    for i, col in enumerate(['size', 'bedrooms', 'age']):
        signo = "+" if coeficientes[i] >= 0 else ""
        ecuacion += f" {signo}{coeficientes[i]:.2f} × {col}"
    
    print(f"   {ecuacion}")
    
    # Crear DataFrame de coeficientes
    coef_df = pd.DataFrame({
        'Variable': ['size', 'bedrooms', 'age'],
        'Coeficiente': coeficientes,
        'Importancia_Absoluta': np.abs(coeficientes)
    }).sort_values('Importancia_Absoluta', ascending=False)
    
    print(f"\n📊 IMPORTANCIA DE LAS VARIABLES:")
    print(coef_df.to_string(index=False))
    
    return modelo, coef_df

def generar_grafico_coeficientes(coef_df, artifacts_dir):
    """
    Generar gráfico de coeficientes
    """
    plt.figure(figsize=(10, 6))
    plt.bar(coef_df['Variable'], coef_df['Coeficiente'], 
           color=['skyblue', 'lightgreen', 'salmon'])
    plt.title('Coeficientes del Modelo de Regresión Lineal', fontsize=14, fontweight='bold')
    plt.xlabel('Variables')
    plt.ylabel('Coeficiente')
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.grid(axis='y', alpha=0.3)
    
    # Agregar valores en las barras
    for i, v in enumerate(coef_df['Coeficiente']):
        plt.text(i, v + (0.01 if v >= 0 else -0.01), f'{v:.2f}', 
                ha='center', va='bottom' if v >= 0 else 'top', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{artifacts_dir}/coeficientes.png', dpi=300, bbox_inches='tight')
    print("✅ Gráfico de coeficientes guardado como 'coeficientes.png'")
    plt.close()

def evaluar_modelo(modelo, X_train, X_test, y_train, y_test, artifacts_dir):
    """
    FASE 5: EVALUACIÓN
    """
    print("\n" + "="*80)
    print("FASE 5: EVALUACIÓN")
    print("="*80)
    
    print("🔮 Generando predicciones...")
    y_pred_train = modelo.predict(X_train)
    y_pred_test = modelo.predict(X_test)
    
    # Calcular métricas
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mae_train = mean_absolute_error(y_train, y_pred_train)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    
    print("📊 MÉTRICAS DE EVALUACIÓN:")
    print(f"   • R² (Entrenamiento): {r2_train:.4f}")
    print(f"   • R² (Prueba): {r2_test:.4f}")
    print(f"   • RMSE (Entrenamiento): {rmse_train:.2f}")
    print(f"   • RMSE (Prueba): {rmse_test:.2f}")
    print(f"   • MAE (Entrenamiento): {mae_train:.2f}")
    print(f"   • MAE (Prueba): {mae_test:.2f}")
    
    print(f"\n💡 INTERPRETACIÓN:")
    print(f"   • R² = {r2_test:.4f} → el modelo explica el {r2_test*100:.1f}% de la variabilidad del precio")
    print(f"   • RMSE = {rmse_test:.2f} → error promedio ≈ {rmse_test:.2f} miles de dólares")
    print(f"   • MAE = {mae_test:.2f} → error absoluto promedio ≈ {mae_test:.2f} miles de dólares")
    
    if r2_test > 0.9:
        print("   🎉 Excelente rendimiento del modelo!")
    elif r2_test > 0.8:
        print("   👍 Buen rendimiento del modelo")
    elif r2_test > 0.7:
        print("   ✅ Rendimiento aceptable del modelo")
    else:
        print("   ⚠️  El modelo podría necesitar mejoras")
    
    # Generar gráficos de evaluación
    generar_graficos_evaluacion(y_test, y_pred_test, artifacts_dir)
    
    return r2_test, rmse_test, mae_test

def generar_graficos_evaluacion(y_test, y_pred_test, artifacts_dir):
    """
    Generar gráficos de evaluación
    """
    print("\n📈 Generando gráficos de evaluación...")
    
    # Predicciones vs Valores reales
    plt.figure(figsize=(10, 8))
    plt.scatter(y_test, y_pred_test, alpha=0.6, color='blue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Precio Real (miles $)')
    plt.ylabel('Precio Predicho (miles $)')
    plt.title('Predicciones vs Valores Reales')
    plt.grid(True, alpha=0.3)
    
    # Agregar R² en el gráfico
    r2 = r2_score(y_test, y_pred_test)
    plt.text(0.05, 0.95, f'R² = {r2:.4f}', transform=plt.gca().transAxes, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'{artifacts_dir}/pred_vs_real.png', dpi=300, bbox_inches='tight')
    print("✅ Gráfico predicciones vs reales guardado como 'pred_vs_real.png'")
    plt.close()
    
    # Residuos
    residuos = y_test - y_pred_test
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Análisis de Residuos', fontsize=16, fontweight='bold')
    
    # Residuos vs Predicciones
    axes[0].scatter(y_pred_test, residuos, alpha=0.6, color='green')
    axes[0].axhline(y=0, color='red', linestyle='--')
    axes[0].set_xlabel('Precio Predicho (miles $)')
    axes[0].set_ylabel('Residuos')
    axes[0].set_title('Residuos vs Predicciones')
    axes[0].grid(True, alpha=0.3)
    
    # Histograma de residuos
    axes[1].hist(residuos, bins=15, alpha=0.7, color='orange', edgecolor='black')
    axes[1].set_xlabel('Residuos')
    axes[1].set_ylabel('Frecuencia')
    axes[1].set_title('Distribución de Residuos')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{artifacts_dir}/residuos.png', dpi=300, bbox_inches='tight')
    print("✅ Gráficos de residuos guardados como 'residuos.png'")
    plt.close()

def guardar_modelo(modelo, scaler, artifacts_dir):
    """
    FASE 6: DESPLIEGUE - Guardar modelo y scaler
    """
    print("\n" + "="*80)
    print("FASE 6: DESPLIEGUE")
    print("="*80)
    
    print("💾 Guardando modelo y scaler...")
    
    # Guardar modelo
    modelo_path = f'{artifacts_dir}/modelo.joblib'
    joblib.dump(modelo, modelo_path)
    print(f"   • Modelo guardado en: {modelo_path}")
    
    # Guardar scaler
    scaler_path = f'{artifacts_dir}/scaler.joblib'
    joblib.dump(scaler, scaler_path)
    print(f"   • Scaler guardado en: {scaler_path}")
    
    print("✅ Modelo y scaler guardados exitosamente!")

def predecir_precio(tamaño, habitaciones, edad, artifacts_dir="./artifacts"):
    """
    Función para predecir el precio de un inmueble
    """
    try:
        # Cargar modelo y scaler
        modelo = joblib.load(f'{artifacts_dir}/modelo.joblib')
        scaler = joblib.load(f'{artifacts_dir}/scaler.joblib')
        
        # Crear array con los datos
        datos_nuevos = np.array([[tamaño, habitaciones, edad]])
        
        # Escalar los datos
        datos_escalados = scaler.transform(datos_nuevos)
        
        # Hacer predicción
        precio_predicho = modelo.predict(datos_escalados)[0]
        
        return precio_predicho
        
    except Exception as e:
        print(f"❌ Error en la predicción: {e}")
        return None

def mostrar_ejemplos_prediccion(artifacts_dir):
    """
    Mostrar ejemplos de predicción
    """
    print("\n🔮 EJEMPLOS DE PREDICCIÓN:")
    ejemplos = [
        (80, 3, 15, "Casa mediana"),
        (120, 5, 5, "Casa grande y nueva"),
        (50, 1, 25, "Apartamento pequeño y viejo"),
        (100, 4, 10, "Casa grande y relativamente nueva")
    ]
    
    for tam, hab, edad, desc in ejemplos:
        precio_pred = predecir_precio(tam, hab, edad, artifacts_dir)
        if precio_pred is not None:
            print(f"   • {desc}: {tam}m², {hab} habitaciones, {edad} años → Precio estimado: ${precio_pred:.0f}k")

def fase7_retroalimentacion():
    """
    FASE 7: RETROALIMENTACIÓN
    """
    print("\n" + "="*80)
    print("FASE 7: RETROALIMENTACIÓN")
    print("="*80)
    
    print("🚀 PRÓXIMOS PASOS RECOMENDADOS:")
    print("   • Validar resultados con el equipo de negocio")
    print("   • Implementar monitoreo continuo del modelo")
    print("   • Reentrenar el modelo periódicamente con nuevos datos")
    print("   • Probar otros algoritmos (Árboles de Decisión, Random Forest, XGBoost)")
    print("   • Considerar variables adicionales (ubicación, amenities, etc.)")
    print("   • Implementar validación cruzada para robustez")
    print("   • Crear dashboard para visualización de predicciones")
    print()

def main():
    """
    Función principal que ejecuta todo el flujo CRISP-DM
    """
    parser = argparse.ArgumentParser(description='CRISP-DM + Regresión Lineal para Predicción de Precios de Inmuebles')
    parser.add_argument('--data', type=str, default='./precios_casa.csv',
                       help='Ruta al archivo CSV de datos (default: ./precios_casa.csv)')
    parser.add_argument('--artifacts', type=str, default='./artifacts',
                       help='Directorio para guardar artefactos (default: ./artifacts)')
    
    args = parser.parse_args()
    
    print("🏠 CRISP-DM + REGRESIÓN LINEAL - PREDICCIÓN DE PRECIOS DE INMUEBLES")
    print("="*80)
    
    # Crear directorio de artefactos
    Path(args.artifacts).mkdir(parents=True, exist_ok=True)
    
    try:
        # Fase 1: Comprensión del negocio
        fase1_comprension_negocio()
        
        # Fase 2: Comprensión de los datos
        df = cargar_datos(args.data)
        if df is None:
            print("❌ No se pudieron cargar los datos. Terminando ejecución.")
            return 1
        
        if not explorar_datos(df):
            print("❌ El dataset no contiene las columnas requeridas. Terminando ejecución.")
            return 1
        
        generar_graficos_exploratorios(df, args.artifacts)
        
        # Fase 3: Preparación de los datos
        X, y = preparar_datos(df)
        X_train, X_test, y_train, y_test = dividir_datos(X, y)
        X_train_scaled, X_test_scaled, scaler = escalar_datos(X_train, X_test)
        
        # Fase 4: Modelado
        modelo, coef_df = entrenar_modelo(X_train_scaled, y_train)
        generar_grafico_coeficientes(coef_df, args.artifacts)
        
        # Fase 5: Evaluación
        r2, rmse, mae = evaluar_modelo(modelo, X_train_scaled, X_test_scaled, y_train, y_test, args.artifacts)
        
        # Fase 6: Despliegue
        guardar_modelo(modelo, scaler, args.artifacts)
        mostrar_ejemplos_prediccion(args.artifacts)
        
        # Fase 7: Retroalimentación
        fase7_retroalimentacion()
        
        print("🎉 ¡PROYECTO COMPLETADO EXITOSAMENTE!")
        print(f"📁 Artefactos guardados en: {args.artifacts}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error durante la ejecución: {e}")
        return 1

if __name__ == "__main__":
    exit(main())

