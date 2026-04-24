# Cargar modelos
# Función predicción
# Predicción
# Resultados

import pandas as pd
import joblib


COLUMNAS = [
    "frecuencia_cardiaca",
    "potencia",
    "cadencia",
    "tiempo",
    "temperatura",
    "pendiente",
    "velocidad"
]


def cargar_modelo(MODELO_PATH):
    return joblib.load(MODELO_PATH)


def predecir(modelo, valores):
    datos = pd.DataFrame([valores], columns=COLUMNAS)
    pred = modelo.predict(datos)
    return float(pred[0])