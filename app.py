import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
from streamlit_option_menu import option_menu
import plotly.express as px
from PIL import Image

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="AI Smart Suite", layout="wide", page_icon="🤖")

# --- COLORES Y ESTILOS VISUALES ---
PRIMARY_COLOR = "#1f77b4"
ACCENT_COLOR = "#ff7f0e"
BACKGROUND_COLOR = "#f8f9fa"
FONT_FAMILY = "Segoe UI, sans-serif"

st.markdown(f"""
    <style>
        body {{
            background-color: {BACKGROUND_COLOR};
            font-family: {FONT_FAMILY};
        }}
        .stApp {{
            background: linear-gradient(135deg, #ffffff, #e6ecf2);
        }}
        .sidebar .sidebar-content {{
            background-color: {PRIMARY_COLOR};
            color: white;
        }}
        .css-1d391kg .stButton>button {{
            background-color: {ACCENT_COLOR};
            color: white;
            font-weight: bold;
            border-radius: 8px;
        }}
        .css-1d391kg .stButton>button:hover {{
            background-color: #e67300;
        }}
    </style>
""", unsafe_allow_html=True)

# --- MENÚ LATERAL ---
with st.sidebar:
    selected = option_menu(
        menu_title="Menú Principal",
        options=[
            "📈 Predicción Demanda", 
            "📂 Análisis de Archivos", 
            "🧠 Análisis de Imágenes",
            "🎥 Video Analytics", 
            "⚙️ Configuración"
        ],
        icons=["graph-up", "file-earmark-text", "image", "camera-video", "gear"],
        default_index=0,
        styles={
            "container": {"padding": "5px", "background-color": PRIMARY_COLOR},
            "icon": {"color": "white", "font-size": "20px"},
            "nav-link": {"font-size": "16px", "--hover-color": ACCENT_COLOR},
            "nav-link-selected": {"background-color": ACCENT_COLOR},
        }
    )

# --- SECCIÓN: Predicción de Demanda ---
def predict_demand_section():
    st.title("📈 Predicción de Demanda con AI")
    st.markdown("""
    Sube un archivo `.csv` con las siguientes columnas:
    - 📅 **fecha** (formato `YYYY-MM-DD`)
    - 📦 **producto**
    - 🔢 **cantidad** (vendida)
    """)

    uploaded_file = st.file_uploader("Sube archivo CSV", type=["csv"])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file, parse_dates=["fecha"])
            df = df.sort_values(["producto", "fecha"])
            st.subheader("Datos cargados")
            st.dataframe(df)

            productos = df["producto"].unique()
            producto_sel = st.selectbox("Selecciona un producto", productos)

            df_producto = df[df["producto"] == producto_sel].copy()

            window = st.slider("Ventana de promedio móvil (días)", 2, 10, 3)
            growth_factor = st.slider("Factor de crecimiento estimado", 1.0, 2.0, 1.1, 0.01)
            forecast_days = st.slider("Días a predecir", 1, 14, 7)

            df_producto["promedio_movil"] = df_producto["cantidad"].rolling(window=window).mean()
            last_avg = df_producto["promedio_movil"].dropna().iloc[-1]

            future_dates = [df_producto["fecha"].iloc[-1] + timedelta(days=i) for i in range(1, forecast_days + 1)]
            forecast_values = [int(round(last_avg * (growth_factor ** i))) for i in range(1, forecast_days + 1)]

            forecast_df = pd.DataFrame({
                "fecha": future_dates,
                "cantidad_predicha": forecast_values
            })

            st.subheader("📊 Pronóstico de demanda")
            combined = pd.concat([
                df_producto.set_index("fecha")["cantidad"],
                forecast_df.set_index("fecha")["cantidad_predicha"]
            ])
            fig = px.line(combined, labels={"index": "Fecha", "value": "Cantidad"}, title="Histórico y Predicción")
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(forecast_df)
        except Exception as e:
            st.error(f"❌ Error al procesar el archivo: {e}")
    else:
        st.info("📤 Por favor, sube un archivo CSV para comenzar.")

# --- SECCIÓN: Análisis de Archivos ---
def file_analysis_section():
    st.title("📂 Análisis Exploratorio de Archivos")
    uploaded_file = st.file_uploader("Carga un archivo CSV para analizar", type=["csv"])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.subheader("📄 Vista previa de los datos")
            st.dataframe(df.head(10))

            st.subheader("📈 Estadísticas básicas")
            st.write(df.describe(include="all"))

            st.subheader("📊 Gráfico de distribución")
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            if numeric_cols:
                col_sel = st.selectbox("Selecciona una columna numérica", numeric_cols)
                fig = px.histogram(df, x=col_sel, nbins=30, title=f"Distribución de {col_sel}")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("⚠️ No se encontraron columnas numéricas.")
        except Exception as e:
            st.error(f"❌ Error procesando archivo: {e}")
    else:
        st.info("📤 Sube un archivo para iniciar el análisis.")

# --- SECCIÓN: Análisis de Imágenes con IA (Simulado) ---
def analyze_image_section():
    st.title("🧠 Análisis Inteligente de Imágenes")
    st.markdown("Sube una imagen y obtén una predicción simulada de su contenido.")

    uploaded_image = st.file_uploader("Sube una imagen (JPG, PNG)", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        st.image(uploaded_image, caption="🖼 Imagen cargada", use_container_width=True)

        with st.spinner("🔍 Analizando imagen..."):
            simulated_labels = ["persona", "taza", "silla", "pantalla", "botella"]
            simulated_confidences = [round(np.random.uniform(0.7, 0.99), 2) for _ in simulated_labels]

            st.subheader("📋 Resultados detectados (simulación):")
            for label, conf in zip(simulated_labels, simulated_confidences):
                st.markdown(f"- **{label.capitalize()}**: {conf * 100:.1f}% de confianza")

            st.success("✅ Análisis completado.")
    else:
        st.info("📤 Por favor, sube una imagen para comenzar.")

# --- SECCIÓN: Placeholder Video ---
def video_analytics_section():
    st.title("🎥 Análisis de Video (próximamente)")
    st.markdown("""
    Esta sección integrará análisis en tiempo real de video usando visión por computador.
    Próximamente se habilitará con modelos de detección de objetos y conteo de personas.
    """)

# --- SECCIÓN: Configuración ---
def settings_section():
    st.title("⚙️ Configuración de la Aplicación")
    st.markdown("""
    Personaliza la experiencia, ajusta parámetros y exporta configuraciones.
    (Esta sección será expandida en futuras versiones).
    """)

# --- ENRUTAMIENTO ---
if selected == "📈 Predicción Demanda":
    predict_demand_section()
elif selected == "📂 Análisis de Archivos":
    file_analysis_section()
elif selected == "🧠 Análisis de Imágenes":
    analyze_image_section()
elif selected == "🎥 Video Analytics":
    video_analytics_section()
elif selected == "⚙️ Configuración":
    settings_section()
