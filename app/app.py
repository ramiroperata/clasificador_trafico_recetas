import joblib 
import json
import logging
import os
from pathlib import Path

import gradio as gr
import pandas as pd

# Configuración del logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Obtener el directorio del script
SCRIPT_DIR = Path.cwd()
MODEL_PATH = SCRIPT_DIR / "recipe_traffic_model.joblib"
scaler_path = SCRIPT_DIR / "recipe_traffic_transformer.joblib"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(scaler_path)

# Leer nombres de las columnas
with open('features.json') as f:
    expected_columns = json.load(f)

# Definir categorías basadas en las columnas
categories = [col.replace('category_', '') for col in expected_columns if col.startswith('category_')]
numerical_cols = ['sugar', 'carbohydrate', 'calories', 'protein', 'servings']

logging.info(f"Categories: {categories}")
logging.info(f"Numerical columns: {numerical_cols}")

def preprocess(input_data):
    """
    Preprocesar los datos de entrada para que coincidan con el formato esperado del modelo.
    """
    logging.debug(f"Raw input data: {input_data}")
    
    try:
        data = pd.DataFrame(0, index=[0], columns=expected_columns)
        
        # Llenar valores numéricos
        for col in numerical_cols:
            data[col] = float(input_data[col])
        # Configurar la categoría
        category_col = f"category_{input_data['category']}"
        if category_col in expected_columns:
            data[category_col] = 1
        
        data[numerical_cols] = scaler.transform(data[numerical_cols])

        return data
    except Exception as e:
        logging.error(f"Error en el preprocesamiento: {str(e)}")
        raise    

def predict(calories, carbohydrate, sugar, protein, category, servings):
    """Hacer predicciones usando el modelo cargado"""
    try:
        # El input_data se mantiene en inglés para evitar problemas en el procesamiento
        input_data = {
            "calories": calories,
            "carbohydrate": carbohydrate,
            "sugar": sugar,
            "protein": protein,
            "servings": servings,
            "category": category  # Debe permanecer en inglés
        }
        
        # Validar entradas
        if any(pd.isna(val) for val in input_data.values() if isinstance(val, (int, float))):
            raise ValueError("Todos los valores numéricos deben ser válidos")
        
        if category not in categories:
            raise ValueError(f"La categoría debe ser una de: {categories}")
        
        processed_data = preprocess(input_data)
        
        prediction = model.predict(processed_data)
        probability = model.predict_proba(processed_data)[:, 1]

        # Resultados estilizados en Markdown
        result = f"""
        ### Resultados de la Predicción:
        - **Tráfico estimado:** {"✅ Alto" if int(prediction[0]) == 1 else "❌ Bajo"}
        - **Probabilidad de tráfico alto:** {float(probability[0]) * 100:.2f}%
        """
        return result

    except Exception as e:
        logging.error(f"Error en la predicción: {str(e)}")
        return f"### Error: {str(e)}"


# Traducción de las categorías para la interfaz, pero manteniendo el orden en inglés para el procesamiento
categories_spanish = [
    "Bebidas", "Desayuno", "Pollo", "Postre", 
    "Merienda", "Carne", "Plato Único", 
    "Cerdo", "Papa", "Vegetal"
]

# Diccionario de traducción español-inglés
category_translation = {
    "Bebidas": "Beverages",
    "Desayuno": "Breakfast",
    "Pollo": "Chicken",
    "Postre": "Dessert",
    "Merienda": "Lunch/Snacks",
    "Carne": "Meat",
    "Plato Único": "One Dish Meal",
    "Cerdo": "Pork",
    "Papa": "Potato",
    "Vegetal": "Vegetable"
}

demo = gr.Interface(
    fn=lambda calories, carbohydrate, sugar, protein, category, servings: predict(
        calories,
        carbohydrate,
        sugar,
        protein,
        category_translation[category],  # Traducir la categoría seleccionada al inglés
        servings
    ),
    inputs=[
        gr.Number(label="Calorías", minimum=0),
        gr.Number(label="Carbohidratos (gramos)", minimum=0),
        gr.Number(label="Azúcar (gramos)", minimum=0),
        gr.Number(label="Proteína (gramos)", minimum=0),
        gr.Dropdown(choices=list(category_translation.keys()), label="Categoría"),  # Mostrar las categorías en español
        gr.Number(label="Porciones", minimum=1)
    ],
    outputs=gr.Markdown(),  # Usar Markdown para un output atractivo
    title="Modelo de predicción de tráfico de recetas",
    description="Ingrese información nutricional y categoría para predecir el tráfico de la receta",
    examples=[
        [300, 40, 5, 15, "Pollo", 4],
        [200, 25, 15, 8, "Postre", 6],
        [400, 50, 2, 30, "Cerdo", 4]
    ]
)

demo.launch()