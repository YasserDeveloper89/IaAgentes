import streamlit as st
from streamlit_option_menu import option_menu
import yaml
import logging

from app.ai_agent import predict_demand
from app.video_analytics import analyze_occupancy, analyze_behavior
from app.utils import setup_logging, validate_config_keys
from app.analytics import load_sales_data, aggregate_sales
import pandas as pd
import plotly.express as px

# Setup logging
setup_logging()

# Load config
def load_config(path: str = "app/config.yaml") -> dict:
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    validate_config_keys(config, ["ai_agent", "video_analytics", "ui"])
    return config

CONFIG = load_config()

st.set_page_config(page_title="IA para Restaurantes y Clínicas", layout="wide")

# CSS styling for menu collapse
st.markdown(
    f"""
    <style>
    html, body, [class*="css"] {{
        font-family: {CONFIG['ui']['font_family']};
    }}
    .stApp {{
        background-color: {CONFIG['ui'].get('secondaryBackgroundColor', '#f0f2f6')};
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar menú con opción desplegable y auto-cierre
with st.sidebar:
    selected = option_menu(
        menu_title="Menú",
        options=[
            "Predicción de Demanda",
            "Análisis de Ocupación y Activos",
            "Análisis de Comportamiento",
            "Dashboard y Alertas"
        ],
        icons=["graph-up", "people-fill", "camera-video", "speedometer2"],
        menu_icon="app-indicator",
        default_index=0,
        orientation="vertical",
        styles={
            "container": {"padding": "5px", "background-color": "#fafafa"},
            "icon": {"color": "#0d6efd", "font-size": "18px"},
            "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
            "nav-link-selected": {"background-color": "#0d6efd", "color": "white"},
        },
    )

# Funciones para cada sección

def section_prediccion_demanda():
    st.header("🔮 Predicción de Demanda Detallada")
    archivo = st.file_uploader("Carga CSV con datos históricos de ventas/inventario", type=["csv"])
    if archivo:
        try:
            df = pd.read_csv(archivo)
            st.dataframe(df.head())
            resultado = predict_demand(df, CONFIG["ai_agent"])
            st.success("Predicción completada")
            st.dataframe(resultado)
        except Exception as e:
            st.error(f"Error: {e}")

def section_analisis_ocupacion():
    st.header("📈 Análisis de Ocupación y Utilización de Activos")
    imagen = st.file_uploader("Sube imagen para análisis de ocupación (JPG, PNG)", type=["jpg","jpeg","png"])
    if imagen:
        try:
            imagen_annotada, count = analyze_occupancy(imagen, CONFIG["video_analytics"])
            st.image(imagen_annotada, caption=f"Personas detectadas: {count}", use_column_width=True)
        except Exception as e:
            st.error(f"Error: {e}")

def section_analisis_comportamiento():
    st.header("🎥 Análisis de Comportamiento de Clientes/Pacientes")
    video = st.file_uploader("Sube video para análisis comportamental (MP4)", type=["mp4"])
    if video:
        try:
            report = analyze_behavior(video, CONFIG["video_analytics"])
            st.write("Resumen de comportamiento detectado:")
            st.json(report)
        except Exception as e:
            st.error(f"Error: {e}")

def section_dashboard_alertas():
    st.header("📊 Dashboard Integrado y Alertas")
    archivo = st.file_uploader("Carga CSV con datos históricos para dashboard", type=["csv"])
    if archivo:
        try:
            df = pd.read_csv(archivo)
            df = load_sales_data(df)
            freq = st.selectbox("Frecuencia para agregación", ["D", "W", "M"], index=0)
            df_agg = aggregate_sales(df, freq)

            fig = px.line(df_agg, x="date", y="quantity", color="product",
                          title="Tendencias Históricas de Ventas", labels={"quantity":"Cantidad","date":"Fecha"})
            st.plotly_chart(fig, use_container_width=True)

            # Aquí se pueden agregar alertas con base en configuración o reglas simples
            threshold = st.number_input("Umbral de alerta para ventas bajas", min_value=0, value=10)
            low_sales = df_agg[df_agg["quantity"] < threshold]
            if not low_sales.empty:
                st.warning(f"Productos con ventas por debajo del umbral ({threshold}):")
                st.dataframe(low_sales)
            else:
                st.success("No hay alertas de ventas bajas.")
        except Exception as e:
            st.error(f"Error: {e}")

# Ejecutar sección según selección
if selected == "Predicción de Demanda":
    section_prediccion_demanda()
elif selected == "Análisis de Ocupación y Activos":
    section_analisis_ocupacion()
elif selected == "Análisis de Comportamiento":
    section_analisis_comportamiento()
elif selected == "Dashboard y Alertas":
    section_dashboard_alertas()
