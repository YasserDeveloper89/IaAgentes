import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
from streamlit_option_menu import option_menu
import plotly.express as px
from ultralytics import YOLO
from PIL import Image

# --- CONFIGURACIÓN DE ESTILO Y PÁGINA ---
BACKGROUND_COLOR = "#0A0A1E"
PRIMARY_COLOR_FUTURISTIC = "#00BCD4"
TEXT_COLOR = "#E0E0E0"
FONT_FAMILY = "Segoe UI, Arial, sans-serif"

st.set_page_config(
    page_title="Plataforma de Herramientas Inteligentes",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🤖"
)

st.markdown(f"""
    <style>
        .stApp {{
            background-color: {BACKGROUND_COLOR};
            color: {TEXT_COLOR};
            font-family: {FONT_FAMILY};
        }}
        .stSidebar {{ background-color: #1A1A30; }}
        h1, h2, h3 {{ color: {PRIMARY_COLOR_FUTURISTIC}; }}
    </style>
""", unsafe_allow_html=True)

# Traducciones para etiquetas detectadas por YOLO
LABEL_TRANSLATIONS = {
    'person': 'Persona',
    'bottle': 'Botella',
    'cup': 'Taza',
    'jeringa': 'Jeringa',
    'mascarilla': 'Mascarilla',
    'guantes medicos': 'Guantes Médicos',
    'fresa': 'Fresa',
    'uva': 'Uva',
    'plato': 'Plato',
    'vaso': 'Vaso'
}

# --- Lógica de selección de tipo de negocio ---
if 'business_type' not in st.session_state:
    st.session_state.business_type = None

if st.session_state.business_type is None:
    st.title("Plataforma de Herramientas Inteligentes para Restaurantes y Clínicas")
    st.markdown("Selecciona el tipo de negocio para comenzar:")

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
    st.sidebar.title(f"Negocio seleccionado: {st.session_state.business_type}")
    if st.sidebar.button("Cambiar negocio"):
        st.session_state.business_type = None
        st.rerun()

    with st.sidebar:
        selected = option_menu(
            menu_title="Herramientas de IA",
            options=["Predicción de Demanda", "Análisis de Archivos", "Análisis de Imágenes", "Configuración"],
            icons=["bar-chart-line", "file-earmark-text", "image", "gear"],
            default_index=0,
            styles={
                "container": {"padding": "5px", "background-color": "#1A1A30"},
                "icon": {"color": PRIMARY_COLOR_FUTURISTIC, "font-size": "20px"},
                "nav-link": {"font-size": "16px", "text-align": "left", "color": TEXT_COLOR},
                "nav-link-selected": {"background-color": PRIMARY_COLOR_FUTURISTIC, "color": "#FFFFFF"}
            }
)
        # --- Herramientas de la plataforma ---

def predict_demand_section():
    st.title("📊 Predicción de Demanda")
    file = st.file_uploader("Sube un archivo CSV con columnas: fecha, elemento, cantidad", type=["csv"])
    if file:
        df = pd.read_csv(file, parse_dates=["fecha"])
        if not all(col in df.columns for col in ["fecha", "elemento", "cantidad"]):
            st.error("El archivo debe contener las columnas: fecha, elemento, cantidad")
            return

        st.dataframe(df)
        elemento = st.selectbox("Selecciona el elemento", df["elemento"].unique())
        df_filtrado = df[df["elemento"] == elemento].sort_values("fecha")

        ventana = st.slider("Tamaño de ventana móvil", 2, 10, 3)
        crecimiento = st.slider("Crecimiento esperado (%)", 0, 100, 5) / 100
        dias = st.slider("Días a predecir", 1, 30, 7)

        df_filtrado["media_movil"] = df_filtrado["cantidad"].rolling(window=ventana).mean()
        valor_base = df_filtrado["media_movil"].dropna().iloc[-1] if not df_filtrado["media_movil"].dropna().empty else df_filtrado["cantidad"].mean()
        fechas = [df_filtrado["fecha"].max() + timedelta(days=i) for i in range(1, dias+1)]
        cantidades = [round(valor_base * (1 + crecimiento) ** i) for i in range(1, dias+1)]
        pred_df = pd.DataFrame({"Fecha": fechas, "Cantidad Prevista": cantidades})

        fig = px.line(pred_df, x="Fecha", y="Cantidad Prevista", title=f"Predicción de Demanda para {elemento}")
        st.plotly_chart(fig)
        st.dataframe(pred_df)

def file_analysis_section():
    st.title("📂 Análisis de Archivos CSV")
    file = st.file_uploader("Sube archivo CSV", type=["csv"])
    if file:
        df = pd.read_csv(file)
        st.subheader("Vista previa de los datos")
        st.dataframe(df.head(10))

        st.subheader("📊 Estadísticas generales")
        descripcion = df.describe(include='all').T
        descripcion.rename(columns={
            "count": "Cantidad",
            "unique": "Valores Únicos",
            "top": "Más Frecuente",
            "freq": "Frecuencia",
            "mean": "Promedio",
            "std": "Desviación Estándar",
            "min": "Mínimo",
            "25%": "Percentil 25%",
            "50%": "Mediana",
            "75%": "Percentil 75%",
            "max": "Máximo"
        }, inplace=True)
        st.dataframe(descripcion)

        columnas_numericas = df.select_dtypes(include=np.number).columns.tolist()
        if columnas_numericas:
            col = st.selectbox("Selecciona una columna numérica", columnas_numericas)
            st.subheader("Distribución (Histograma)")
            st.plotly_chart(px.histogram(df, x=col, nbins=30, title=f"Distribución de {col}"))
            st.subheader("Valores Atípicos (Boxplot)")
            st.plotly_chart(px.box(df, y=col, title=f"Diagrama de Caja de {col}"))

def image_analysis_section():
    st.title("📸 Análisis Inteligente de Imágenes")
    modelo = st.radio("Modelo de detección", ["YOLOv8 General", "YOLO-World"])

    # Objetos sugeridos por tipo de negocio
    default_objetos = ""
    if st.session_state.business_type == "Restaurante":
        default_objetos = "fresa, uva, plátano, empanada, pizza, plato, vaso, cuchillo, tenedor"
    elif st.session_state.business_type == "Clínica":
        default_objetos = "jeringa, guantes médicos, mascarilla, termómetro, camilla, medicamento"

    objetos = st.text_input("📝 Objetos personalizados a detectar (separados por coma)", value=default_objetos)
    archivo = st.file_uploader("Sube una imagen (JPG, PNG)", type=["jpg", "jpeg", "png"])
    if archivo:
        imagen = Image.open(archivo)
        st.image(imagen, caption="Imagen original", use_container_width=True)
        modelo_yolo = YOLO("yolov8n.pt" if modelo == "YOLOv8 General" else "yolov8s-world.pt")
        if modelo == "YOLO-World" and objetos.strip():
            modelo_yolo.set_classes([c.strip().lower() for c in objetos.split(",") if c.strip()])
        resultado = modelo_yolo(imagen)[0]
        st.image(resultado.plot(), caption="Resultado IA", use_container_width=True)

        boxes = resultado.boxes.data.cpu().numpy()
        nombres = resultado.names
        datos = []
        for box in boxes:
            x1, y1, x2, y2, score, clase = box
            etiqueta = nombres[int(clase)]
            nombre_es = LABEL_TRANSLATIONS.get(etiqueta.lower(), etiqueta)
            datos.append({
                "Objeto Detectado": nombre_es,
                "Confianza": f"{score * 100:.2f}%",
                "Coordenadas": f"[{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}]"
            })
        if datos:
            st.subheader("📋 Objetos Detectados")
            st.dataframe(pd.DataFrame(datos))
        else:
            st.info("No se detectaron objetos con el modelo seleccionado.")

def settings_section():
    st.title("⚙️ Configuración")
    st.info("Aquí podrás configurar preferencias personalizadas en futuras versiones.")

# --- Ruteo según selección ---
if st.session_state.business_type:
    if selected == "Predicción de Demanda":
        predict_demand_section()
    elif selected == "Análisis de Archivos":
        file_analysis_section()
    elif selected == "Análisis de Imágenes":
        image_analysis_section()
    elif selected == "Configuración":
        settings_section()
