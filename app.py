import streamlit as st import pandas as pd import numpy as np from datetime import timedelta from streamlit_option_menu import option_menu import plotly.express as px from PIL import Image import tempfile from ultralytics import YOLO import torch

--- CONFIGURACIÓN VISUAL ---

PRIMARY_COLOR = "#1f77b4" ACCENT_COLOR = "#ff7f0e" BACKGROUND_COLOR = "#f0f2f6" FONT_FAMILY = "Arial, sans-serif"

st.set_page_config(page_title="AI Agents + Video Analytics", layout="wide", page_icon="🤖")

--- ESTILOS CSS PARA ESTÉTICA MODERNA ---

st.markdown(f""" <style> .stApp {{ background: linear-gradient(135deg, #e0eafc 0%, #cfdef3 100%); color: #333; font-family: {FONT_FAMILY}; }} .sidebar .sidebar-content {{ background-color: {PRIMARY_COLOR}; color: white; }} .css-1d391kg .stButton>button {{ background-color: {ACCENT_COLOR}; color: white; border-radius: 8px; padding: 8px 16px; font-weight: bold; }} .css-1d391kg .stButton>button:hover {{ background-color: #e67300; color: white; }} </style> """, unsafe_allow_html=True)

--- MENU LATERAL ---

with st.sidebar: selected = option_menu( menu_title="Menu Principal", options=["Predicción Demanda", "Análisis de Archivos", "Análisis Imagen", "Configuración"], icons=["bar-chart-line", "file-earmark-text", "image", "gear"], menu_icon="cast", default_index=0, styles={ "container": {"padding": "5px", "background-color": PRIMARY_COLOR}, "icon": {"color": "white", "font-size": "20px"}, "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": ACCENT_COLOR}, "nav-link-selected": {"background-color": ACCENT_COLOR}, } )

--- FUNCIONES ---

def predict_demand_section(): st.title("📊 Predicción de Demanda") st.markdown(""" Carga un archivo CSV con las siguientes columnas: - 📅 fecha (formato YYYY-MM-DD) - 📦 producto - 🔢 cantidad (vendida) """) uploaded_file = st.file_uploader("Sube archivo CSV", type=["csv"]) if uploaded_file: try: df = pd.read_csv(uploaded_file, parse_dates=['fecha']) df = df.sort_values(['producto', 'fecha']) st.subheader("📄 Vista Previa de Datos") st.dataframe(df)

productos = df['producto'].unique()
        producto_sel = st.selectbox("Selecciona el producto a analizar", productos)

        df_producto = df[df['producto'] == producto_sel].copy()

        window = st.slider("Tamaño ventana de promedio móvil (días)", 2, 10, 3)
        growth_factor = st.slider("Factor de crecimiento estimado", 1.0, 2.0, 1.1, 0.01)
        forecast_days = st.slider("Cantidad de días a predecir", 1, 14, 7)

        df_producto['moving_avg'] = df_producto['cantidad'].rolling(window=window).mean()
        last_avg = df_producto['moving_avg'].dropna().iloc[-1]

        future_dates = [df_producto['fecha'].max() + timedelta(days=i) for i in range(1, forecast_days + 1)]
        forecast_values = [round(last_avg * (growth_factor ** i)) for i in range(1, forecast_days + 1)]

        forecast_df = pd.DataFrame({'fecha': future_dates, 'prediccion_cantidad': forecast_values})

        st.subheader(f"📈 Predicción para: {producto_sel}")
        combined = pd.concat([
            df_producto.set_index('fecha')['cantidad'],
            forecast_df.set_index('fecha')['prediccion_cantidad']
        ])
        fig = px.line(combined, labels={'index':'Fecha', 'value':'Cantidad'}, title="Demanda Histórica y Predicción")
        st.plotly_chart(fig, use_container_width=True)
        st.write(forecast_df)

    except Exception as e:
        st.error(f"Error al procesar el archivo: {e}")
else:
    st.info("Sube un archivo CSV para comenzar el análisis.")

def file_analysis_section(): st.title("📁 Análisis Inteligente de Archivos") st.markdown("Explora tus archivos de datos y obtén estadísticas y visualizaciones automáticas.") uploaded_file = st.file_uploader("Sube archivo CSV", type=["csv"]) if uploaded_file: try: df = pd.read_csv(uploaded_file) st.subheader("👁 Vista previa de los primeros datos") st.dataframe(df.head())

st.subheader("📊 Estadísticas Generales")
        st.write(df.describe(include='all'))

        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if numeric_cols:
            col_sel = st.selectbox("Selecciona una columna para graficar", numeric_cols)
            fig = px.histogram(df, x=col_sel, nbins=30, title=f"Distribución de {col_sel}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No hay columnas numéricas para graficar.")
    except Exception as e:
        st.error(f"Error procesando el archivo: {e}")
else:
    st.info("Por favor, sube un archivo para comenzar.")

def image_analysis_section(): st.title("🧠 Análisis de Imagen con IA") st.markdown("Sube una imagen para detectar objetos y personas automáticamente usando visión por computador.") uploaded_image = st.file_uploader("Sube una imagen (JPG, PNG)", type=["jpg", "jpeg", "png"]) if uploaded_image: try: model = YOLO("yolov8n.pt") with tempfile.NamedTemporaryFile(delete=False) as temp_file: temp_file.write(uploaded_image.read()) temp_path = temp_file.name

results = model(temp_path)[0]
        img = Image.open(temp_path)
        boxes = results.boxes

        st.image(img, caption="Imagen Original", use_container_width=True)

        if boxes:
            st.subheader("📝 Resultados de detección:")
            for box in boxes:
                cls = int(box.cls[0])
                name = model.names[cls]
                st.markdown(f"- Objeto: **{name}**, Confianza: {float(box.conf[0]):.2f}")
        else:
            st.info("No se detectaron objetos en la imagen.")
    except Exception as e:
        st.error(f"Ocurrió un error procesando la imagen: {e}")
else:
    st.info("Por favor, sube una imagen para comenzar.")

def settings_section(): st.title("⚙️ Configuración General") st.markdown("Desde aquí podrás modificar ajustes generales en futuras versiones de la app.")

--- RUTEO DE SECCIONES ---

if selected == "Predicción Demanda": predict_demand_section() elif selected == "Análisis de Archivos": file_analysis_section() elif selected == "Análisis Imagen": image_analysis_section() elif selected == "Configuración": settings_section() else: st.write("Selecciona una opción del menú lateral.")

