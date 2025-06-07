import streamlit as st
import yaml
import os
from app.ai_agent import load_and_validate_sales, predict_demand
from app.data_analysis import analyze_uploaded_file

# Cargar configuración
CONFIG_PATH = os.path.join('app', 'config.yaml')
with open(CONFIG_PATH) as f:
    CONFIG = yaml.safe_load(f)

# CSS para estilo moderno y fondo
st.markdown(f"""
<style>
    .reportview-container {{
        background-color: {CONFIG['ui']['background_color']};
        font-family: {CONFIG['ui']['font_family']};
    }}
    .sidebar .sidebar-content {{
        background-color: {CONFIG['ui']['secondary_background_color']};
    }}
    .css-1aumxhk {{
        padding-top: 1rem;
    }}
</style>
""", unsafe_allow_html=True)

# Menú lateral desplegable
with st.sidebar:
    st.title("IA Agentes + Video Analytics")
    section = st.selectbox("Seleccione sección", 
                           ["Predicción de Demanda", "Análisis Inteligente de Archivos"])

st.title(section)

if section == "Predicción de Demanda":
    st.subheader("Cargue archivo CSV con columnas 'fecha' y 'ventas'")

    uploaded_file = st.file_uploader("Archivo de ventas CSV", type=["csv"])
    if uploaded_file:
        try:
            df = load_and_validate_sales(uploaded_file)
            st.dataframe(df)

            if st.button("Generar Predicción"):
                result_df, fig = predict_demand(df, CONFIG)
                st.plotly_chart(fig, use_container_width=True)
                st.write(result_df.tail(10))
        except Exception as e:
            st.error(f"Error: {e}")

elif section == "Análisis Inteligente de Archivos":
    st.subheader("Cargue archivo CSV para análisis automático")

    uploaded_file = st.file_uploader("Archivo CSV para análisis", type=["csv"])
    if uploaded_file:
        try:
            summary, fig = analyze_uploaded_file(uploaded_file)
            st.write("Resumen Estadístico:")
            st.dataframe(summary)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error: {e}")
