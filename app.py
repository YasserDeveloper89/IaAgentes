import streamlit as st
from streamlit_option_menu import option_menu
import yaml
from pathlib import Path

# Carga de configuración
CONFIG_PATH = Path(__file__).parent / "config.yaml"
with open(CONFIG_PATH, "r") as f:
    CONFIG = yaml.safe_load(f)

# Estilos dinámicos desde config.yaml
st.set_page_config(
    page_title="IA Agentes Inteligentes",
    layout="wide",
    initial_sidebar_state="auto",
)

st.markdown(
    f"""
    <style>
    body {{
        background-color: {CONFIG['ui']['secondaryBackgroundColor']};
        font-family: {CONFIG['ui']['font_family']};
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# Menú desplegable lateral
with st.sidebar:
    selected = option_menu(
        menu_title="Menú principal",
        options=["Inicio", "Predicción de Demanda", "Análisis de Archivos"],
        icons=["house", "graph-up", "folder"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "5px", "background-color": "#f9f9f9"},
            "icon": {"color": CONFIG['ui']['accent_color'], "font-size": "20px"},
            "nav-link": {
                "font-size": "16px",
                "text-align": "left",
                "margin": "5px",
                "--hover-color": "#eee",
            },
            "nav-link-selected": {
                "background-color": CONFIG['ui']['primary_color'],
                "color": "white",
            },
        },
    )

# INICIO
if selected == "Inicio":
    st.title("📊 IA Agentes Inteligentes")
    st.markdown("Bienvenido. Usa el menú lateral para acceder a herramientas de predicción, análisis y más.")

# PREDICCIÓN DE DEMANDA
elif selected == "Predicción de Demanda":
    st.title("📈 Predicción de Demanda")
    from app.ai_agent import predict_demand

    uploaded_file = st.file_uploader("Sube un archivo CSV con datos históricos", type=["csv"])
    if uploaded_file:
        predict_demand(uploaded_file, CONFIG)

# ANÁLISIS DE ARCHIVOS
elif selected == "Análisis de Archivos":
    st.title("🧠 Análisis Inteligente de Archivos")
    from app.data_analysis import analyze_uploaded_file

    uploaded_file = st.file_uploader("Sube un archivo CSV o Excel", type=["csv", "xlsx", "xls"])
    if uploaded_file:
        analyze_uploaded_file(uploaded_file)
