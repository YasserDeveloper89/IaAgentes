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

# --- ESTILOS CSS PARA ESTÉTICA MODERNA ---
st.markdown(f"""
    <style>
        .stApp {{
            background: linear-gradient(135deg, #e0eafc 0%, #cfdef3 100%);
            color: #333;
            font-family: {FONT_FAMILY};
        }}
        .sidebar .sidebar-content {{
            background-color: {PRIMARY_COLOR};
            color: white;
        }}
        .stButton>button {{
            background-color: {ACCENT_COLOR};
            color: white;
            border-radius: 8px;
            padding: 8px 16px;
            font-weight: bold;
        }}
        .stButton>button:hover {{
            background-color: #e67300;
            color: white;
        }}
        .stSelectbox>div>div {{
            color: #333;
        }}
        .stDataFrame, .stTable {{
            border-radius: 10px;
            background-color: white;
        }}
    </style>
""", unsafe_allow_html=True)

# --- MENÚ LATERAL CON streamlit-option-menu ---
with st.sidebar:
    selected = option_menu(
        menu_title="Menú Principal",
        options=["📈 Predicción de Demanda", "📂 Análisis de Archivos", "🎥 Video Analytics", "⚙️ Configuración"],
        icons=["graph-up-arrow", "folder", "camera-video", "gear"],
        default_index=0,
        styles={
            "container": {"padding": "5px", "background-color": PRIMARY_COLOR},
            "icon": {"color": "white", "font-size": "18px"},
            "nav-link": {"font-size": "16px", "color": "white", "--hover-color": ACCENT_COLOR},
            "nav-link-selected": {"background-color": ACCENT_COLOR},
        }
    )

# --- SECCIÓN: PREDICCIÓN DE DEMANDA ---
def predict_demand_section():
    st.title("📈 Predicción de Demanda")
    st.markdown("""
    Sube un archivo CSV con las siguientes columnas:
    - **fecha** (formato YYYY-MM-DD)
    - **producto**
    - **cantidad** (vendida)
    """)

    uploaded_file = st.file_uploader("📤 Sube archivo CSV", type=["csv"])

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file, parse_dates=["fecha"])
            df = df.sort_values(["producto", "fecha"])

            st.subheader("🔎 Vista Previa de Datos")
            st.dataframe(df)

            productos = df["producto"].unique()
            producto_sel = st.selectbox("📌 Selecciona un producto para predecir", productos)

            df_producto = df[df["producto"] == producto_sel].copy()

            window = st.slider("📊 Días para promedio móvil", 2, 10, 3)
            growth_factor = st.slider("📈 Factor de crecimiento", 1.0, 2.0, 1.1, step=0.05)
            forecast_days = st.slider("📅 Días a predecir", 1, 30, 7)

            df_producto["moving_avg"] = df_producto["cantidad"].rolling(window=window).mean()
            last_avg = df_producto["moving_avg"].dropna().iloc[-1]

            future_dates = [df_producto["fecha"].iloc[-1] + timedelta(days=i) for i in range(1, forecast_days + 1)]
            forecast_values = [round(last_avg * (growth_factor ** i)) for i in range(1, forecast_days + 1)]

            forecast_df = pd.DataFrame({
                "fecha": future_dates,
                "predicción_cantidad": forecast_values
            })

            st.subheader(f"📉 Pronóstico de Demanda para: {producto_sel}")
            combined = pd.concat([
                df_producto.set_index("fecha")["cantidad"],
                forecast_df.set_index("fecha")["predicción_cantidad"]
            ], axis=1)

            fig = px.line(combined, title="Histórico y Predicción", labels={"value": "Cantidad", "fecha": "Fecha"})
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("📋 Resultados del Pronóstico")
            st.table(forecast_df)

        except Exception as e:
            st.error(f"❌ Error al procesar el archivo: {e}")
    else:
        st.info("☝️ Carga un archivo CSV para comenzar.")

# --- SECCIÓN: ANÁLISIS DE ARCHIVOS ---
def file_analysis_section():
    st.title("📂 Análisis Inteligente de Archivos")
    st.markdown("Sube un archivo CSV para visualizarlo, explorar columnas y obtener estadísticas descriptivas.")

    uploaded_file = st.file_uploader("📤 Sube archivo CSV", type=["csv"])

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)

            st.subheader("👁️ Vista previa")
            st.dataframe(df.head(10))

            st.subheader("📊 Estadísticas Básicas")
            st.write(df.describe(include='all'))

            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            if numeric_cols:
                col_sel = st.selectbox("📈 Selecciona columna numérica para graficar", numeric_cols)
                fig = px.histogram(df, x=col_sel, nbins=30, title=f"Distribución de {col_sel}")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("⚠️ No hay columnas numéricas para graficar.")

        except Exception as e:
            st.error(f"❌ Error procesando archivo: {e}")
    else:
        st.info("☝️ Sube un archivo para comenzar el análisis.")

# --- SECCIÓN: VIDEO ANALYTICS (SIMULADA) ---
def video_analytics_section():
    st.title("🎥 Video Analytics (Demo)")
    st.markdown("""
    Aquí se mostraría el análisis de comportamiento de clientes, ocupación de espacios, etc.  
    Esta demo no incluye procesamiento de video real para evitar dependencias como OpenCV.
    """)

    st.success("✅ Próximamente se integrarán análisis en tiempo real de cámaras.")

# --- SECCIÓN: CONFIGURACIÓN ---
def settings_section():
    st.title("⚙️ Configuración")
    st.markdown("""
    Opciones de ajuste generales de la aplicación.  
    Aquí podrías añadir personalización de colores, guardar configuraciones, etc.
    """)

# --- RUTEO POR SECCIÓN ---
if selected == "📈 Predicción de Demanda":
    predict_demand_section()
elif selected == "📂 Análisis de Archivos":
    file_analysis_section()
elif selected == "🎥 Video Analytics":
    video_analytics_section()
elif selected == "⚙️ Configuración":
    settings_section()
