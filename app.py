import os
import joblib
import streamlit as st

from src.train import entrenar_modelos
from src.test import cargar_modelo, predecir

DATASET_PATH = "data/dataset_ciclismo_fatiga.csv"

MODELO_LR_PATH = "models/modelo_lr.pkl"
MODELO_KNN_PATH = "models/modelo_knn.pkl"
MODELO_ARBOL_PATH = "models/modelo_arbol.pkl"
METRICAS_PATH = "models/metricas.pkl"

st.set_page_config(page_title="Sistema de Fatiga", layout="centered")

st.title("Sistema de Machine Learning - Fatiga")

# =========================
# MENÚ
# =========================
opcion = st.radio(
    "Selecciona una opción",
    ("Entrenar modelos", "Predecir"),
    horizontal=True
)

# =========================
# ENTRENAMIENTO
# =========================
if opcion == "Entrenar modelos":

    st.markdown("## 🔧 Entrenamiento de modelos")

    with st.container():
        st.info(
            "Se entrenan 3 modelos:\n"
            "- Regresión Lineal (con estandarización)\n"
            "- KNN (con estandarización)\n"
            "- Árbol de Decisión (sin estandarización)"
        )

    if st.button("Entrenar los 3 modelos"):

        try:
            metricas, mejor_modelo = entrenar_modelos(
                DATASET_PATH,
                MODELO_LR_PATH,
                MODELO_KNN_PATH,
                MODELO_ARBOL_PATH,
                METRICAS_PATH
            )

            st.success("Modelos entrenados correctamente")

            st.markdown("## 📊 Comparación de modelos")

            col1, col2, col3 = st.columns(3)

            modelos_lista = list(metricas.keys())
            columnas = [col1, col2, col3]

            for i, nombre in enumerate(modelos_lista):
                with columnas[i]:
                    st.markdown(f"### {nombre}")
                    st.metric("MSE", f"{metricas[nombre]['MSE']:.2f}")
                    st.metric("MAE", f"{metricas[nombre]['MAE']:.2f}")
                    st.metric("R2", f"{metricas[nombre]['R2']:.4f}")

            st.markdown("---")
            st.success(f"🏆 Mejor modelo: {mejor_modelo}")

        except Exception as e:
            st.error(f"Error: {e}")

# =========================
# PREDICCIÓN
# =========================
elif opcion == "Predecir":

    st.markdown("## 🔍 Predicción de fatiga")

    with st.container():
        st.markdown("### 📥 Ingreso de datos")

        col1, col2 = st.columns(2)

        with col1:
            fc = st.number_input("Frecuencia cardíaca", value=150.0)
            pot = st.number_input("Potencia", value=250.0)
            cad = st.number_input("Cadencia", value=90.0)
            tiempo = st.number_input("Tiempo", value=45.0)

        with col2:
            temp = st.number_input("Temperatura", value=30.0)
            pend = st.number_input("Pendiente", value=5.0)
            vel = st.number_input("Velocidad", value=28.0)

    if st.button("Predecir con los 3 modelos"):

        rutas = {
            "Regresión Lineal": MODELO_LR_PATH,
            "KNN": MODELO_KNN_PATH,
            "Árbol de Decisión": MODELO_ARBOL_PATH
        }

        if not os.path.exists(METRICAS_PATH):
            st.warning("Primero debes entrenar los modelos")
        else:
            try:
                valores_entrada = [fc, pot, cad, tiempo, temp, pend, vel]
                metricas = joblib.load(METRICAS_PATH)

                resultados = {}

                for nombre, ruta in rutas.items():
                    modelo = cargar_modelo(ruta)
                    resultados[nombre] = predecir(modelo, valores_entrada)

                st.markdown("## 📈 Resultados de predicción")

                col1, col2, col3 = st.columns(3)

                for i, nombre in enumerate(resultados.keys()):
                    with [col1, col2, col3][i]:
                        st.metric(nombre, f"{resultados[nombre]:.2f}")

                st.markdown("---")

                st.markdown("## 🧠 Comparación de modelos")

                for nombre, valores in metricas.items():
                    with st.expander(f"Ver métricas de {nombre}"):
                        st.write(f"MSE: {valores['MSE']:.2f}")
                        st.write(f"MAE: {valores['MAE']:.2f}")
                        st.write(f"R2: {valores['R2']:.4f}")

                mejor_modelo = max(metricas, key=lambda m: metricas[m]["R2"])

                st.markdown("---")

                st.success(f"🏆 Mejor modelo: {mejor_modelo}")
                st.info(f"Predicción recomendada: {resultados[mejor_modelo]:.2f}")

            except Exception as e:
                st.error(f"Error: {e}")