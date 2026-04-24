# Cargar datos
# Dividir datos
# Pipeline
# Entrenar
# Evaluar
# Guardar modelos

import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def entrenar_modelos(DATASET_PATH, MODELO_LR_PATH, MODELO_KNN_PATH, MODELO_ARBOL_PATH, METRICAS_PATH):

    data = pd.read_csv(DATASET_PATH)

    X = data.drop("fatiga", axis=1)
    y = data["fatiga"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    modelo_lr = Pipeline([
        ("scaler", StandardScaler()),
        ("modelo", LinearRegression())
    ])

    modelo_knn = Pipeline([
        ("scaler", StandardScaler()),
        ("modelo", KNeighborsRegressor(n_neighbors=9))
    ])

    modelo_arbol = Pipeline([
        ("modelo", DecisionTreeRegressor(max_depth=5, random_state=42))
    ])

    modelos = {
        "Regresión Lineal": modelo_lr,
        "KNN": modelo_knn,
        "Árbol de Decisión": modelo_arbol
    }

    rutas = {
        "Regresión Lineal": MODELO_LR_PATH,
        "KNN": MODELO_KNN_PATH,
        "Árbol de Decisión": MODELO_ARBOL_PATH
    }

    metricas = {}

    for nombre, modelo in modelos.items():
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)

        metricas[nombre] = {
            "MSE": mean_squared_error(y_test, y_pred),
            "MAE": mean_absolute_error(y_test, y_pred),
            "R2": r2_score(y_test, y_pred)
        }

        joblib.dump(modelo, rutas[nombre])

    mejor_modelo = max(metricas, key=lambda m: metricas[m]["R2"])

    joblib.dump(metricas, METRICAS_PATH)

    return metricas, mejor_modelo