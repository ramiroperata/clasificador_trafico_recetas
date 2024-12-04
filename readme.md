# 🥘 Clasificador de Tráfico de Recetas  

Este repositorio contiene un proyecto de ciencia de datos enfocado en clasificar recetas según su nivel de tráfico utilizando modelos de aprendizaje automático. Desde la recolección y limpieza de datos hasta la implementación de un modelo de clasificación, este proyecto proporciona una solución integral.  

*[Probalo acá](https://huggingface.co/spaces/ramiropm/recipe_traffic_classifier)*

## 🛠️ Componentes del Proyecto  
1. **Recolección y Procesamiento de Datos**  
   - Obtención de datos de recetas y su tráfico.  
   - Limpieza y preprocesamiento para garantizar la calidad.  

2. **Entrenamiento del Modelo**  
   - Modelos de clasificación para predecir recetas de alto o bajo tráfico.  
   - Optimización de hiperparámetros y evaluación.  

3. **Despliegue de la Aplicación**  
   - Aplicación interactiva construida con Gradio.  
   - Permite al usuario cargar datos de recetas y obtener predicciones en tiempo real.  

## 📂 Estructura del Repositorio  
```
📁 clasificador_trafico_recetas
├── 📂 data/ # Datos originales
├── 📂 notebook/ # Notebook de análisis y entrenamiento
├── 📂 app/ # Código y dependencias de la aplicación
│ ├── app.py # Código de la aplicación Gradio
│ ├── requirements.txt # Dependencias de la aplicación
├── 📂 model/ # Modelo entrenado
├── 📜 README.md # Este archivo
└── 📜 LICENSE # Información sobre la licencia
```
## 🔍 Exploración y Análisis de Datos  
Los datos de recetas fueron obtenidos de una fuente pública, analizados para identificar patrones relevantes y preprocesados para alimentar el modelo. En la carpeta `notebooks/` encontrarás:  
- Visualización de distribución de tráfico.  
- Selección de características clave para la predicción.  

## 🧠 Entrenamiento del Modelo  
El modelo principal clasifica recetas en categorías de alto y bajo tráfico. Pasos destacados:  
1. Preparación del dataset con etiquetas claras.  
2. Entrenamiento y evaluación de 2 modelos `Random Forest` y `Logistic Regression`.  
3. Selección del modelo final basado en precisión y accuracy.  

## 🚀 Despliegue de la Aplicación  
La aplicación interactiva permite al usuario explorar el clasificador de tráfico en un entorno accesible y visual.  

Consulta el [README de la aplicación](app/README.md) para más detalles.  

## 📊 Resultados  
El modelo alcanzó una precisión del 82%, intentando asegurar que las recetas que predecimos como alto trafico efectivamente lo son. Los detalles de la evaluación se encuentran documentados en los notebooks de la carpeta `notebook/`.  

## 🤝 Contribuciones  
Contribuciones y sugerencias son bienvenidas. Por favor, abre un **issue** o envía un **pull request**.  

## 📜 Licencia  
Este proyecto está bajo la licencia Apache 2.0. Consulta el archivo [LICENSE](LICENSE) para más detalles.  
