import streamlit as st
import pandas as pd
import yaml
from app.ai_agent import predict_demand
from app.video_analytics import analyze_occupancy
from app.utils import setup_logging, validate_config_keys

# Configuración global
setup_logging()

# Cargar configuración desde YAML
def load_config(path: str = "app/config.yaml") -> dict:
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    required_sections = ["ai_agent", "video_analytics", "ui"]
    validate_config_keys(config, required_sections)
    return config

CONFIG = load_config()
st.set_page_config(page_title="IA Agente + Video", layout="wide")

# Estilo UI
st.markdown(
    f"""
    <style>
    html, body, [class*="css"] {{
        font-family: {CONFIG['ui']['font_family']};
    }}
    .stApp {{
        background-color: {CONFIG['ui']['secondaryBackgroundColor']};
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# Interfaz
st.title("📊 Agente de IA + Análisis de Video")
st.sidebar.title("Funciones")

opcion = st.sidebar.selectbox("Selecciona una función", ["Predicción de Demanda", "Análisis de Ocupación"])

if opcion == "Predicción de Demanda":
    st.subheader("🔮 Predicción de demanda por insumo")

    archivo = st.file_uploader("Sube archivo CSV con ventas históricas", type=["csv"])
    if archivo:
        try:
            df = pd.read_csv(archivo)
            st.dataframe(df.head())

            resultado = predict_demand(df, CONFIG["ai_agent"])
            st.success("Predicción completada")
            st.dataframe(resultado)

        except Exception as e:
            st.error(f"Error procesando archivo: {e}")

elif opcion == "Análisis de Ocupación":
    st.subheader("🎥 Detección de personas y ocupación")
    imagen = st.file_uploader("Sube una imagen (JPG, PNG)", type=["jpg", "jpeg", "png"])
    if imagen:
        try:
            imagen_anotada, count = analyze_occupancy(imagen, CONFIG["video_analytics"])
            st.image(imagen_anotada, caption=f"Personas detectadas: {count}", use_column_width=True)
        except Exception as e:
            st.error(f"Error procesando imagen: {e}")
