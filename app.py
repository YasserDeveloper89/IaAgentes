import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
from streamlit_option_menu import option_menu
import plotly.express as px
from ultralytics import YOLO
from PIL import Image
import io

# --- CONFIGURACIÓN VISUAL Y DE TEMAS ---
BACKGROUND_COLOR = "#0A0A1E"
PRIMARY_COLOR_FUTURISTIC = "#00BCD4"
ACCENT_COLOR_FUTURISTIC = "#FF4081"
TEXT_COLOR = "#E0E0E0"
SECONDARY_TEXT_COLOR = "#A0A0B0"
BORDER_COLOR = "#2C2C40"
FONT_FAMILY = "Segoe UI, Arial, sans-serif"

st.set_page_config(
    page_title="Plataforma de IA Corporativa: Soluciones Avanzadas",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="✨"
)

st.markdown(f"""
<style>
/* Aquí va tu CSS completo como el original */
body {{ background-color: {BACKGROUND_COLOR}; color: {TEXT_COLOR}; }}
</style>
""", unsafe_allow_html=True)

LABEL_TRANSLATIONS = {{
    'person': 'Persona', 'bottle': 'Botella', 'cup': 'Taza', 'chair': 'Silla',
    'jeringa': 'Jeringa', 'mascarilla': 'Mascarilla', 'guantes medicos': 'Guantes Médicos',
    'fresa': 'Fresa', 'uva': 'Uva', 'plato': 'Plato', 'vaso': 'Vaso'
}}

# --- SELECCIÓN DEL TIPO DE NEGOCIO ---
if 'business_type' not in st.session_state:
    st.session_state.business_type = None

business_options = {{
    "Restaurante": "🍽️ Soluciones para la gestión culinaria y de clientes.",
    "Clínica": "🏥 Optimización de procesos sanitarios y atención al paciente."
}}

if st.session_state.business_type is None:
    st.title("Bienvenido a la Plataforma de IA Corporativa")
    st.markdown("Selecciona tu tipo de negocio para personalizar la experiencia.")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Soy un Restaurante"):
            st.session_state.business_type = "Restaurante"
            st.rerun()
    with col2:
        if st.button("Soy una Clínica"):
            st.session_state.business_type = "Clínica"
            st.rerun()
else:
    st.sidebar.title(f"Tipo de Negocio: {st.session_state.business_type}")
    if st.sidebar.button("Cambiar tipo de negocio"):
        st.session_state.business_type = None
        st.rerun()

    with st.sidebar:
        selected = option_menu(
            menu_title="Módulos de IA",
            options=["Predicción Demanda", "Análisis Archivos", "Análisis de Imágenes", "Configuración"],
            icons=["bar-chart-line", "file-earmark-text", "image", "gear"],
            default_index=0,
            styles={{"icon": {{"color": PRIMARY_COLOR_FUTURISTIC}}}}
)
        # --- Funciones de las herramientas (deben ir antes del ruteo del menú) ---

def predict_demand_section():
    st.title("📊 Predicción de Demanda")
    uploaded_file = st.file_uploader("Sube CSV con columnas: fecha, elemento, cantidad", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file, parse_dates=["fecha"])
        st.dataframe(df)
        elementos = df["elemento"].unique()
        selected_element = st.selectbox("Selecciona elemento", elementos)
        df_elem = df[df["elemento"] == selected_element].sort_values("fecha")
        window = st.slider("Ventana media móvil", 2, 10, 3)
        growth = st.slider("Factor crecimiento", 1.0, 2.0, 1.05)
        days = st.slider("Días a predecir", 1, 30, 7)

        df_elem["media_movil"] = df_elem["cantidad"].rolling(window).mean()
        last = df_elem["media_movil"].dropna().iloc[-1]
        forecast = [round(last * (growth**i)) for i in range(1, days+1)]
        fechas = [df_elem["fecha"].iloc[-1] + timedelta(days=i) for i in range(1, days+1)]
        df_pred = pd.DataFrame({"fecha": fechas, "cantidad": forecast})

        fig = px.line(df_pred, x="fecha", y="cantidad", title="Predicción de Demanda")
        st.plotly_chart(fig)
        st.dataframe(df_pred)

def file_analysis_section():
    st.title("📂 Análisis de Archivos CSV")
    file = st.file_uploader("Sube archivo CSV", type=["csv"])
    if file:
        df = pd.read_csv(file)
        st.dataframe(df.head(10))
        st.write("Estadísticas:", df.describe())
        num_cols = df.select_dtypes(include=np.number).columns
        if len(num_cols):
            col = st.selectbox("Columna para histograma", num_cols)
            st.plotly_chart(px.histogram(df, x=col))

def image_analysis_section():
    st.title("📸 Detección de Objetos en Imágenes")
    model_type = st.radio("Tipo de modelo", ["YOLOv8 General", "YOLO-World"])
    objects = st.text_input("Objetos personalizados (si usas YOLO-World)", "")
    image = st.file_uploader("Sube una imagen", type=["jpg", "png", "jpeg"])
    if image:
        img = Image.open(image)
        st.image(img, caption="Imagen subida", use_container_width=True)
        model = YOLO("yolov8n.pt" if model_type == "YOLOv8 General" else "yolov8s-world.pt")
        if model_type == "YOLO-World" and objects:
            model.set_classes([o.strip() for o in objects.split(",")])
        results = model(img)
        st.image(results[0].plot(), caption="Resultado", use_container_width=True)

        boxes = results[0].boxes.data.cpu().numpy()
        labels = results[0].names
        data = []
        for box in boxes:
            x1, y1, x2, y2, score, cls = box
            name = labels[int(cls)]
            trad = LABEL_TRANSLATIONS.get(name.lower(), name)
            data.append({
                "Objeto": trad,
                "Confianza": f"{score:.2%}",
                "Coordenadas": f"{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}"
            })
        st.dataframe(pd.DataFrame(data))

def settings_section():
    st.title("⚙️ Configuración")
    st.info("Aquí se agregarán futuras configuraciones.")

# --- Ruteo de secciones ---
if st.session_state.business_type:
    if selected == "Predicción Demanda":
        predict_demand_section()
    elif selected == "Análisis Archivos":
        file_analysis_section()
    elif selected == "Análisis de Imágenes":
        image_analysis_section()
    elif selected == "Configuración":
        settings_section()
