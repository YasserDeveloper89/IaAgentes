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

# Estilos CSS
st.markdown(f"""
    <style>
        .stApp {{
            background-color: {BACKGROUND_COLOR};
            color: {TEXT_COLOR};
            font-family: {FONT_FAMILY};
        }}
        .stSidebar {{
            background-color: #1A1A30;
        }}
        h1, h2, h3 {{
            color: {PRIMARY_COLOR_FUTURISTIC};
        }}
    </style>
""", unsafe_allow_html=True)

# Traducciones para detecciones con YOLO
LABEL_TRANSLATIONS = {
    'person': 'Persona', 'bottle': 'Botella', 'cup': 'Taza',
    'jeringa': 'Jeringa', 'mascarilla': 'Mascarilla', 'guantes medicos': 'Guantes Médicos',
    'fresa': 'Fresa', 'uva': 'Uva', 'plato': 'Plato', 'vaso': 'Vaso'
}

# Selección del tipo de negocio
if 'business_type' not in st.session_state:
    st.session_state.business_type = None

business_options = {
    "Restaurante": "🍽️ Soluciones para la gestión culinaria y de clientes.",
    "Clínica": "🏥 Optimización de procesos sanitarios y atención al paciente."
}

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
            styles={
                "container": {"padding": "5px", "background-color": "#1A1A30"},
                "icon": {"color": PRIMARY_COLOR_FUTURISTIC, "font-size": "20px"},
                "nav-link": {"font-size": "16px", "text-align": "left", "color": TEXT_COLOR},
                "nav-link-selected": {"background-color": PRIMARY_COLOR_FUTURISTIC, "color": "#FFFFFF"},
            }
        )
        # --- Herramientas de la plataforma ---

def predict_demand_section():
    st.title("📊 Predicción de Demanda")
    uploaded_file = st.file_uploader("Sube CSV con columnas: fecha, elemento, cantidad", type=["csv"])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file, parse_dates=["fecha"])
            if "elemento" not in df.columns or "cantidad" not in df.columns:
                st.error("Tu archivo debe tener columnas: fecha, elemento, cantidad.")
                return
            st.dataframe(df)
            elementos = df["elemento"].unique()
            selected = st.selectbox("Selecciona elemento", elementos)
            df_elem = df[df["elemento"] == selected].sort_values("fecha")

            window = st.slider("Ventana media móvil", 2, 10, 3)
            growth = st.slider("Crecimiento (%)", 0, 100, 5) / 100
            days = st.slider("Días a predecir", 1, 30, 7)

            df_elem["media_movil"] = df_elem["cantidad"].rolling(window).mean()
            last_val = df_elem["media_movil"].dropna().iloc[-1] if not df_elem["media_movil"].dropna().empty else df_elem["cantidad"].mean()
            forecast = [round(last_val * (1 + growth) ** i) for i in range(1, days + 1)]
            fechas = [df_elem["fecha"].max() + timedelta(days=i) for i in range(1, days + 1)]
            pred_df = pd.DataFrame({"fecha": fechas, "cantidad": forecast})

            fig = px.line(pred_df, x="fecha", y="cantidad", title="Predicción de Demanda")
            st.plotly_chart(fig)
            st.dataframe(pred_df)

        except Exception as e:
            st.error(f"Error procesando archivo: {e}")

def file_analysis_section():
    st.title("📂 Análisis de Archivos CSV")
    file = st.file_uploader("Sube archivo CSV", type=["csv"])
    if file:
        df = pd.read_csv(file)
        st.dataframe(df.head(10))
        st.write("Estadísticas:", df.describe())
        num_cols = df.select_dtypes(include=np.number).columns
        if len(num_cols):
            col = st.selectbox("Columna numérica para analizar", num_cols)
            st.plotly_chart(px.histogram(df, x=col))

def image_analysis_section():
    st.title("📸 Detección de Objetos en Imágenes")
    model_type = st.radio("Modelo", ["YOLOv8 General", "YOLO-World"])
    objetos = st.text_input("Objetos personalizados (solo para YOLO-World)", "")
    image = st.file_uploader("Sube imagen (JPG, PNG)", type=["jpg", "jpeg", "png"])
    if image:
        img = Image.open(image)
        st.image(img, caption="Imagen original", use_container_width=True)
        model = YOLO("yolov8n.pt" if model_type == "YOLOv8 General" else "yolov8s-world.pt")
        if model_type == "YOLO-World" and objetos.strip():
            model.set_classes([o.strip().lower() for o in objetos.split(",") if o.strip()])
        results = model(img)
        res_img = results[0].plot()
        st.image(res_img, caption="Resultado IA", use_container_width=True)

        detections = results[0].boxes.data.cpu().numpy()
        labels = results[0].names
        data = []
        for box in detections:
            x1, y1, x2, y2, score, cls = box
            etiqueta = labels[int(cls)]
            traducida = LABEL_TRANSLATIONS.get(etiqueta.lower(), etiqueta)
            data.append({
                "Objeto": traducida,
                "Confianza": f"{score*100:.2f}%",
                "Coordenadas": f"[{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}]"
            })
        if data:
            st.dataframe(pd.DataFrame(data))
        else:
            st.info("No se detectaron objetos.")

def settings_section():
    st.title("⚙️ Configuración")
    st.info("Aquí podrás ajustar preferencias y configuraciones futuras.")

# --- Ruteo según opción seleccionada ---
if st.session_state.business_type:
    if selected == "Predicción Demanda":
        predict_demand_section()
    elif selected == "Análisis Archivos":
        file_analysis_section()
    elif selected == "Análisis de Imágenes":
        image_analysis_section()
    elif selected == "Configuración":
        settings_section()
