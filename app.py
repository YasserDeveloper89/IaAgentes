import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
from streamlit_option_menu import option_menu
import plotly.express as px
from PIL import Image
import tempfile
from ultralytics import YOLO
import torch

# --- CONFIGURACIÓN VISUAL ---
PRIMARY_COLOR = "#1f77b4"
ACCENT_COLOR = "#ff7f0e"
BACKGROUND_COLOR = "#f0f2f6"
FONT_FAMILY = "Arial, sans-serif"

st.set_page_config(page_title="AI Agents + Video Analytics", layout="wide", page_icon="🤖")

# --- ESTILOS CSS ---
st.markdown(f"""
    <style>
        .stApp {{
            background: linear-gradient(135deg, #e0eafc 0%, #cfdef3 100%);
            font-family: {FONT_FAMILY};
            color: #333;
        }}
        .sidebar .sidebar-content {{
            background-color: {PRIMARY_COLOR};
            color: white;
        }}
        .css-1d391kg .stButton>button {{
            background-color: {ACCENT_COLOR};
            color: white;
            border-radius: 8px;
            padding: 8px 16px;
        }}
    </style>
""", unsafe_allow_html=True)

# --- MENÚ LATERAL ---
with st.sidebar:
    selected = option_menu(
        menu_title="Menu Principal",
        options=["Predicción Demanda", "Análisis de Archivos", "Análisis de Imagen", "Configuración"],
        icons=["bar-chart-line", "file-earmark-text", "image", "gear"],
        menu_icon="cast",
        default_index=0,
    )

# --- FUNCIONES ---
def predict_demand_section():
    st.title("📊 Predicción de Demanda")
    st.markdown("""
    Carga un CSV con las columnas:
    - 🗓️ **fecha** (formato YYYY-MM-DD)
    - 📦 **producto**
    - 🔢 **cantidad** (vendida)
    """)
    uploaded_file = st.file_uploader("Sube archivo CSV", type=["csv"])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file, parse_dates=['fecha'])
            df = df.sort_values(['producto', 'fecha'])
            st.subheader("Datos cargados")
            st.dataframe(df)

            productos = df['producto'].unique()
            producto_sel = st.selectbox("Selecciona un producto para predecir demanda", productos)

            df_producto = df[df['producto'] == producto_sel].copy()

            window = st.slider("Ventana para promedio móvil (en días)", 2, 10, 3)
            growth_factor = st.slider("Factor de crecimiento estimado", 1.0, 2.0, 1.1, 0.01)
            forecast_days = st.slider("Cantidad de días a predecir", 1, 14, 7)

            df_producto['moving_avg'] = df_producto['cantidad'].rolling(window=window).mean()
            last_avg = df_producto['moving_avg'].iloc[-1]

            future_dates = [df_producto['fecha'].iloc[-1] + timedelta(days=i) for i in range(1, forecast_days + 1)]
            forecast_values = [int(round(last_avg * (growth_factor ** i))) for i in range(1, forecast_days + 1)]

            forecast_df = pd.DataFrame({'fecha': future_dates, 'Demanda estimada': forecast_values})

            st.subheader("Pronóstico")
            combined = pd.concat([
                df_producto.set_index('fecha')['cantidad'],
                forecast_df.set_index('fecha')['Demanda estimada']
            ], axis=1)
            fig = px.line(combined, labels={'value': 'Cantidad', 'fecha': 'Fecha'}, title="Histórico vs Predicción")
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(forecast_df)

        except Exception as e:
            st.error(f"Error al procesar el archivo: {e}")
    else:
        st.info("Carga un archivo CSV para comenzar.")

def file_analysis_section():
    st.title("📂 Análisis de Archivos")
    uploaded_file = st.file_uploader("Sube un archivo CSV para analizar", type=["csv"])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.subheader("Vista previa")
            st.dataframe(df.head())

            st.subheader("Estadísticas generales")
            st.write(df.describe(include='all'))

            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            if numeric_cols:
                col_sel = st.selectbox("Selecciona una columna numérica para visualizar", numeric_cols)
                fig = px.histogram(df, x=col_sel, nbins=20, title=f"Distribución de {col_sel}")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No se encontraron columnas numéricas para graficar.")
        except Exception as e:
            st.error(f"Error procesando archivo: {e}")
    else:
        st.info("Sube un archivo CSV para comenzar el análisis.")

def image_analysis_section():
    st.title("🧠 Análisis Inteligente de Imagen")
    st.markdown("""
    Carga una imagen para detectar personas y objetos mediante IA.
    """)
    uploaded_image = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_image.read())
            img_path = tmp_file.name

        model = YOLO("yolov8n.pt")
        results = model(img_path)
        res = results[0]

        boxes = res.boxes
        img = Image.open(img_path)
        st.image(res.plot(), caption="Resultado del Análisis", use_column_width=True)

        names = res.names if hasattr(res, 'names') else model.names

        if boxes:
            st.subheader("Objetos Detectados")
            for box in boxes:
                cls = int(box.cls[0].item())
                label = names[cls] if cls in names else f"Clase {cls}"
                conf = box.conf[0].item()
                st.write(f"- **{label}** con confianza del {conf:.2%}")
        else:
            st.info("No se detectaron objetos en la imagen.")

def settings_section():
    st.title("⚙️ Configuración")
    st.write("Personalización de parámetros e información de la app")

# --- SELECCIÓN DE SECCIÓN ---
if selected == "Predicción Demanda":
    predict_demand_section()
elif selected == "Análisis de Archivos":
    file_analysis_section()
elif selected == "Análisis de Imagen":
    image_analysis_section()
elif selected == "Configuración":
    settings_section()
                    
