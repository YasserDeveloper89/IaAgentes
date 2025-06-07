import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
from streamlit_option_menu import option_menu
import plotly.express as px

# --- CONFIGURACIÓN VISUAL ---
PRIMARY_COLOR = "#1f77b4"
ACCENT_COLOR = "#ff7f0e"
BACKGROUND_COLOR = "#f0f2f6"
FONT_FAMILY = "Arial, sans-serif"

st.set_page_config(page_title="AI Agents + Video Analytics", layout="wide", page_icon="🤖")

# --- ESTILO PERSONALIZADO ---
st.markdown(f"""
    <style>
        .stApp {{
            background: linear-gradient(135deg, #e0eafc 0%, #cfdef3 100%);
            font-family: {FONT_FAMILY};
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
            font-weight: bold;
        }}
        .css-1d391kg .stButton>button:hover {{
            background-color: #e67300;
        }}
    </style>
""", unsafe_allow_html=True)

# --- MENU LATERAL ---
with st.sidebar:
    selected = option_menu(
        menu_title="Menú Principal",
        options=["📊 Predicción Demanda", "🔍 Análisis de Archivos", "🎥 Video Analytics", "⚙️ Configuración"],
        icons=["graph-up-arrow", "file-bar-graph", "camera-video", "gear"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "5px", "background-color": PRIMARY_COLOR},
            "icon": {"color": "white", "font-size": "18px"},
            "nav-link": {
                "font-size": "16px",
                "color": "white",
                "text-align": "left",
                "margin": "0px",
                "--hover-color": ACCENT_COLOR
            },
            "nav-link-selected": {"background-color": ACCENT_COLOR}
        }
    )

# --- SECCIÓN: PREDICCIÓN DE DEMANDA ---
def predict_demand_section():
    st.title("📊 Predicción de Demanda")
    st.markdown("""
    Carga un archivo CSV con las siguientes columnas obligatorias:
    - `fecha` (formato YYYY-MM-DD)
    - `producto` (nombre del producto)
    - `cantidad` (unidades vendidas)
    """)

    uploaded_file = st.file_uploader("📁 Sube archivo CSV", type=["csv"])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file, parse_dates=['fecha'])
            df = df.sort_values(['producto', 'fecha'])
            st.success("✅ Archivo cargado exitosamente")
            st.dataframe(df.head(10))

            productos = df['producto'].unique()
            producto_sel = st.selectbox("Selecciona un producto para predecir su demanda", productos)

            df_producto = df[df['producto'] == producto_sel].copy()
            window = st.slider("Ventana de promedio móvil (días)", 2, 10, 3)
            growth_factor = st.slider("Factor de crecimiento esperado", 1.0, 2.0, 1.1, 0.01)
            forecast_days = st.slider("Cantidad de días a predecir", 1, 30, 7)

            df_producto['moving_avg'] = df_producto['cantidad'].rolling(window=window).mean()
            last_avg = df_producto['moving_avg'].dropna().iloc[-1]

            future_dates = [df_producto['fecha'].max() + timedelta(days=i) for i in range(1, forecast_days + 1)]
            forecast_values = [last_avg * (growth_factor ** i) for i in range(1, forecast_days + 1)]

            forecast_df = pd.DataFrame({
                'fecha': future_dates,
                'cantidad_predicha': forecast_values
            })

            st.subheader("📈 Resultados del pronóstico")
            combined = pd.concat([
                df_producto.set_index('fecha')['cantidad'],
                forecast_df.set_index('fecha')['cantidad_predicha']
            ], axis=1)
            combined.columns = ['Histórico', 'Pronóstico']

            fig = px.line(combined, title=f"Demanda histórica y pronóstico para: {producto_sel}",
                          labels={"value": "Cantidad", "fecha": "Fecha"})
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(forecast_df)

        except Exception as e:
            st.error(f"❌ Error al procesar el archivo: {e}")
    else:
        st.info("📌 Por favor, sube un archivo para comenzar.")

# --- SECCIÓN: ANÁLISIS DE ARCHIVOS ---
def file_analysis_section():
    st.title("🔍 Análisis Inteligente de Archivos")
    st.markdown("Carga un archivo CSV para visualizar información clave y estadísticas de forma sencilla.")

    uploaded_file = st.file_uploader("📁 Sube archivo CSV para análisis", type=["csv"])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("✅ Archivo cargado")
            st.subheader("📄 Vista previa de los datos")
            st.dataframe(df.head(10))

            st.subheader("📊 Resumen estadístico")
            st.write(df.describe(include='all'))

            st.subheader("📈 Visualización de columnas numéricas")
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            if numeric_cols:
                col_sel = st.selectbox("Selecciona columna numérica para graficar", numeric_cols)
                fig = px.histogram(df, x=col_sel, nbins=30, title=f"Distribución de valores: {col_sel}")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("⚠️ No se encontraron columnas numéricas para graficar.")

        except Exception as e:
            st.error(f"❌ Error procesando archivo: {e}")
    else:
        st.info("📌 Sube un archivo CSV para iniciar el análisis.")

# --- SECCIÓN: VIDEO ANALYTICS (DEMO) ---
def video_analytics_section():
    st.title("🎥 Video Analytics (Demo)")
    st.markdown("""
    Esta sección simula inteligencia visual mediante análisis de cámaras.  
    En la versión final se integrará análisis real de ocupación y comportamiento a partir de video.
    """)
    st.success("✅ Módulo en desarrollo para integración futura de CV y análisis de movimiento en tiempo real.")

# --- SECCIÓN: CONFIGURACIÓN ---
def settings_section():
    st.title("⚙️ Configuración")
    st.markdown("""
    Aquí podrás modificar ajustes de visualización y parámetros de los modelos en futuras versiones.
    """)

# --- RUTEO ---
if selected == "📊 Predicción Demanda":
    predict_demand_section()
elif selected == "🔍 Análisis de Archivos":
    file_analysis_section()
elif selected == "🎥 Video Analytics":
    video_analytics_section()
elif selected == "⚙️ Configuración":
    settings_section()
