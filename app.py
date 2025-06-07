import streamlit as st
import yaml
from app.ai_agent import predict_demand
from app.data_analysis import analyze_uploaded_file

# Cargar configuración
CONFIG_PATH = "config.yaml"
with open(CONFIG_PATH) as f:
    CONFIG = yaml.safe_load(f)

# Configuración general de la página
st.set_page_config(
    page_title="AI Agents + Video Analytics",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Estilo CSS para fondo y fuente moderna
st.markdown(f"""
    <style>
    body {{
        background-color: {CONFIG['ui'].get('secondaryBackgroundColor', '#f0f2f6')};
        font-family: {CONFIG['ui'].get('font_family', 'Arial, sans-serif')};
        color: {CONFIG['ui'].get('primary_color', '#1f77b4')};
        margin: 0 3%;
    }}
    .stButton>button {{
        background-color: {CONFIG['ui'].get('accent_color', '#ff7f0e')};
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: bold;
    }}
    </style>
""", unsafe_allow_html=True)

st.title("Aplicación AI Agents + Video Analytics")

# Menú desplegable avanzado en sidebar
menu = st.sidebar.selectbox(
    "Selecciona la sección:",
    [
        "Predicción de demanda",
        "Análisis de ocupación",
        "Análisis de comportamiento",
        "Análisis inteligente de archivos"
    ]
)

if menu == "Predicción de demanda":
    st.header("Predicción de demanda")
    uploaded_file = st.file_uploader("Carga archivo CSV de ventas", type=["csv"])
    if uploaded_file:
        try:
            results = predict_demand(uploaded_file, CONFIG)
            st.success("Predicción completada!")
            st.dataframe(results)
        except Exception as e:
            st.error(f"Error en predicción: {e}")

elif menu == "Análisis de ocupación":
    st.header("Análisis de ocupación")
    uploaded_video = st.file_uploader("Carga video para análisis de ocupación", type=["mp4", "avi"])
    if uploaded_video:
        try:
            occupancy_report = analyze_occupancy(uploaded_video, CONFIG)
            st.success("Análisis de ocupación completado!")
            st.write(occupancy_report)
        except Exception as e:
            st.error(f"Error en análisis de ocupación: {e}")

elif menu == "Análisis de comportamiento":
    st.header("Análisis de comportamiento del cliente")
    uploaded_video = st.file_uploader("Carga video para análisis de comportamiento", type=["mp4", "avi"])
    if uploaded_video:
        try:
            behavior_report = analyze_behavior(uploaded_video, CONFIG)
            st.success("Análisis de comportamiento completado!")
            st.write(behavior_report)
        except Exception as e:
            st.error(f"Error en análisis de comportamiento: {e}")

elif menu == "Análisis inteligente de archivos":
    st.header("Análisis inteligente de archivos")
    uploaded_file = st.file_uploader("Carga archivo para análisis", type=["csv", "xlsx", "xls"])
    if uploaded_file:
        try:
            file_analysis = analyze_uploaded_file(uploaded_file)
            st.success("Análisis de archivo completado!")
            st.dataframe(file_analysis)
        except Exception as e:
            st.error(f"Error en análisis de archivo: {e}")
