import streamlit as st
from app.ai_agent import predict_demand
from app.data_analysis import analyze_uploaded_file, advanced_summary
import pandas as pd

st.set_page_config(page_title="AI Agents + Video Analytics", layout="wide", page_icon="🤖")

# --- CSS para dark mode y estilo corporativo ---
st.markdown("""
<style>
body {
    background-color: #121212;
    color: #e0e0e0;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
h1, h2, h3 {
    color: #00bcd4;
}
.sidebar .sidebar-content {
    background-image: linear-gradient(#0d47a1, #1976d2);
    color: white;
}
.stButton>button {
    background-color: #00bcd4;
    color: #121212;
    font-weight: bold;
    border-radius: 8px;
    padding: 8px 16px;
}
.stButton>button:hover {
    background-color: #008ba3;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# --- Menú desplegable ---
menu = st.sidebar.selectbox("Selecciona la sección:", [
    "Predicción de Demanda",
    "Análisis Inteligente de Archivos",
    "Resumen Avanzado"
])

st.title("🤖 AI Agents + Video Analytics")

if menu == "Predicción de Demanda":
    st.header("Predicción de Demanda para Restaurantes/Clínicas")
    uploaded_file = st.file_uploader("Carga tu archivo CSV de ventas", type=["csv"])
    if uploaded_file:
        try:
            prediction_df = predict_demand(uploaded_file, config={
                "ai_agent": {"growth_factor": 1.1}
            })
            st.success("Predicción generada exitosamente:")
            st.dataframe(prediction_df)
        except Exception as e:
            st.error(f"Error: {e}")

elif menu == "Análisis Inteligente de Archivos":
    st.header("Análisis Inteligente de Archivos CSV")
    uploaded_file = st.file_uploader("Carga tu archivo CSV para análisis", type=["csv"])
    if uploaded_file:
        try:
            report = analyze_uploaded_file(uploaded_file)
            st.success("Análisis básico generado:")
            st.dataframe(report)
        except Exception as e:
            st.error(f"Error: {e}")

elif menu == "Resumen Avanzado":
    st.header("Resumen Avanzado con Visualización Interactiva")
    uploaded_file = st.file_uploader("Carga tu archivo CSV para resumen", type=["csv"])
    if uploaded_file:
        try:
            fig = advanced_summary(uploaded_file)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error: {e}")
