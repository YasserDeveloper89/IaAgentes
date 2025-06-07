# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
from app.ai_agent import predict_demand
from app.video_analytics import analyze_occupancy
from app.utils import setup_logging
import yaml

setup_logging()

# Carga configuración externa con colores y branding
with open("app/config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

PRIMARY_COLOR = config.get("ui", {}).get("primary_color", "#1f77b4")
ACCENT_COLOR = config.get("ui", {}).get("accent_color", "#ff7f0e")
FONT_FAMILY = config.get("ui", {}).get("font_family", "Arial, sans-serif")

st.set_page_config(
    page_title="AI Agent + Video Analytics",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS para personalización básica
st.markdown(
    f"""
    <style>
        .css-18e3th9 {{ padding-top: 1rem; }}
        .block-container {{ padding: 1rem 2rem 2rem 2rem; }}
        .stButton > button {{ background-color: {PRIMARY_COLOR}; color: white; border-radius: 8px; }}
        .stButton > button:hover {{ background-color: {ACCENT_COLOR}; color: white; }}
        .sidebar .sidebar-content {{
            background-color: #f5f5f5;
            padding-top: 2rem;
            font-family: {FONT_FAMILY};
        }}
        h1, h2, h3 {{
            font-family: {FONT_FAMILY};
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar con logo y menú
with st.sidebar:
    st.image("https://raw.githubusercontent.com/tu_usuario/tu_repo/main/logo.png", width=180)
    st.markdown("## Menú")
    menu = st.radio(
        "",
        ["Predicción de Demanda", "Análisis de Video"],
        index=0,
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        """
        <small style="color:#999;">
        © 2025 AI Solutions · Version 1.0
        </small>
        """,
        unsafe_allow_html=True,
    )

st.title("🧠 AI Agent para Restaurantes y Clínicas")

if menu == "Predicción de Demanda":
    st.header("📦 Predicción de Insumos por Demanda")

    uploaded_file = st.file_uploader(
        "Sube archivo CSV de facturas o ventas",
        type=["csv"],
        help="El archivo debe contener columnas de producto, cantidad y fecha.",
    )

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.subheader("Vista previa de datos:")
            st.dataframe(df.head())

            if st.button("Calcular Predicción"):
                with st.spinner("Procesando predicción..."):
                    resultado = predict_demand(df, config["ai_agent"])

                st.success("Demanda estimada por insumo:")

                # Mostrar tabla con paginación
                st.dataframe(resultado.style.format("{:.2f}"))

                # Gráfico interactivo de demanda por insumo
                fig = px.bar(
                    resultado,
                    x="insumo",
                    y="cantidad_estimda",
                    title="Demanda estimada por insumo",
                    labels={"cantidad_estimda": "Cantidad estimada", "insumo": "Insumo"},
                    color="cantidad_estimda",
                    color_continuous_scale="Viridis",
                )
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error leyendo el archivo: {e}")

elif menu == "Análisis de Video":
    st.header("🎥 Análisis de Ocupación (Demo)")

    image_file = st.file_uploader(
        "Sube imagen para análisis de ocupación",
        type=["jpg", "jpeg", "png"],
        help="Se analizará el número de personas en la imagen."
    )

    if image_file:
        try:
            with st.spinner("Analizando imagen..."):
                image, count = analyze_occupancy(image_file, config["video_analytics"])

            st.image(image, caption=f"Personas detectadas: {count}", use_column_width=True)
            st.success(f"Ocupación estimada: {count} personas")

        except Exception as e:
            st.error(f"Error analizando imagen: {e}")

else:
    st.info("Selecciona una opción en el menú lateral para comenzar.")
