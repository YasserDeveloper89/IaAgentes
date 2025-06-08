import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
from streamlit_option_menu import option_menu
import plotly.express as px
from ultralytics import YOLO
from PIL import Image

# --- Configuración de la página y estilo visual ---
st.set_page_config(
    page_title="Plataforma de herramientas inteligentes",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🧠"
)

st.markdown("""
    <style>
        .stApp {
            background-color: #0A0A1E;
            color: #E0E0E0;
            font-family: 'Segoe UI', sans-serif;
        }
        .stSidebar {
            background-color: #1A1A30;
        }
        h1, h2, h3 {
            color: #00BCD4;
        }
    </style>
""", unsafe_allow_html=True)

# Traducción de etiquetas de objetos
LABEL_TRANSLATIONS = {
    'person': 'Persona', 'bottle': 'Botella', 'cup': 'Taza',
    'jeringa': 'Jeringa', 'mascarilla': 'Mascarilla', 'guantes medicos': 'Guantes Médicos',
    'fresa': 'Fresa', 'uva': 'Uva', 'plato': 'Plato', 'vaso': 'Vaso'
}

# Selección del tipo de negocio
if 'business_type' not in st.session_state:
    st.session_state.business_type = None

if st.session_state.business_type is None:
    st.title("Plataforma de herramientas inteligentes para restaurantes y clínicas")
    st.markdown("Seleccione el tipo de negocio para comenzar a utilizar las herramientas disponibles.")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Restaurante"):
            st.session_state.business_type = "Restaurante"
            st.rerun()
    with col2:
        if st.button("Clínica"):
            st.session_state.business_type = "Clínica"
            st.rerun()
else:
    st.sidebar.title(f"Negocio seleccionado: {st.session_state.business_type}")
    if st.sidebar.button("Cambiar tipo de negocio"):
        st.session_state.business_type = None
        st.rerun()

    with st.sidebar:
        selected = option_menu(
            menu_title="Herramientas de IA",
            options=["Predicción de demanda", "Análisis de archivos", "Análisis de imágenes", "Configuración"],
            icons=["bar-chart-line", "file-earmark-text", "image", "gear"],
            default_index=0,
            styles={
                "container": {"padding": "5px", "background-color": "#1A1A30"},
                "icon": {"color": "#00BCD4", "font-size": "20px"},
                "nav-link": {"font-size": "16px", "text-align": "left", "color": "#E0E0E0"},
                "nav-link-selected": {"background-color": "#00BCD4", "color": "#FFFFFF"}
            }
        )
        # --- Herramientas disponibles ---

def predict_demand_section():
    st.title("Predicción de demanda de productos")
    st.markdown("Esta herramienta permite estimar la demanda futura de un producto a partir de datos históricos.")

    file = st.file_uploader("Seleccione un archivo CSV con columnas: fecha, elemento, cantidad", type=["csv"])
    if file:
        df = pd.read_csv(file, parse_dates=["fecha"])
        if not all(col in df.columns for col in ["fecha", "elemento", "cantidad"]):
            st.error("El archivo debe contener las columnas: fecha, elemento y cantidad.")
            return

        st.subheader("Vista previa de datos")
        st.dataframe(df)

        elemento = st.selectbox("Seleccione el producto a analizar", df["elemento"].unique())
        df_filtro = df[df["elemento"] == elemento].sort_values("fecha")

        ventana = st.slider(
            "Tamaño de ventana móvil",
            2, 10, 3,
            help="Cantidad de días recientes usados para calcular el promedio base de demanda."
        )
        crecimiento = st.slider(
            "Tasa de crecimiento estimada (%)",
            0, 100, 5,
            help="Porcentaje diario de crecimiento esperado en la demanda futura."
        ) / 100
        dias = st.slider(
            "Días a predecir",
            1, 30, 7,
            help="Número de días hacia el futuro para los que se desea realizar la predicción."
        )

        df_filtro["media_movil"] = df_filtro["cantidad"].rolling(window=ventana).mean()
        base = df_filtro["media_movil"].dropna().iloc[-1] if not df_filtro["media_movil"].dropna().empty else df_filtro["cantidad"].mean()
        fechas = [df_filtro["fecha"].max() + timedelta(days=i) for i in range(1, dias+1)]
        cantidades = [round(base * (1 + crecimiento) ** i) for i in range(1, dias+1)]
        pred_df = pd.DataFrame({"Fecha": fechas, "Cantidad Prevista": cantidades})

        st.subheader("Proyección estimada")
        fig = px.line(pred_df, x="Fecha", y="Cantidad Prevista", title=f"Predicción de Demanda: {elemento}")
        st.plotly_chart(fig)
        st.dataframe(pred_df)

def file_analysis_section():
    st.title("Análisis de archivos CSV")
    st.markdown("Este módulo permite visualizar y explorar datos tabulares almacenados en archivos CSV.")

    file = st.file_uploader("Seleccione un archivo CSV", type=["csv"])
    if file:
        df = pd.read_csv(file)
        st.subheader("Vista previa")
        st.dataframe(df.head(10))

        st.subheader("Estadísticas descriptivas")
        desc = df.describe(include='all').T
        desc.rename(columns={
            "count": "Cantidad",
            "unique": "Valores únicos",
            "top": "Valor más frecuente",
            "freq": "Frecuencia",
            "mean": "Promedio",
            "std": "Desviación Estándar",
            "min": "Mínimo",
            "25%": "Percentil 25",
            "50%": "Mediana",
            "75%": "Percentil 75",
            "max": "Máximo"
        }, inplace=True)
        st.dataframe(desc)

        columnas_num = df.select_dtypes(include=np.number).columns.tolist()
        if columnas_num:
            col = st.selectbox("Seleccione una columna numérica para análisis visual", columnas_num)
            st.subheader("Distribución")
            st.plotly_chart(px.histogram(df, x=col, nbins=30, title=f"Histograma de {col}"))
            st.subheader("Análisis de valores extremos")
            st.plotly_chart(px.box(df, y=col, title=f"Boxplot de {col}"))

def image_analysis_section():
    st.title("Análisis de imágenes con detección de objetos")
    st.markdown("Detecta objetos en imágenes utilizando modelos de inteligencia artificial.")

    modelo = st.radio("Modelo de detección", ["YOLOv8 General", "YOLO-World"])

    objetos_default = ""
    if st.session_state.business_type == "Restaurante":
        objetos_default = "fresa, uva, empanada, plato, cuchillo, vaso"
    elif st.session_state.business_type == "Clínica":
        objetos_default = "face mask, syringe, thermometer, hospital bed, medicine bottle"

    objetos = st.text_input(
        "Lista de objetos personalizados (solo para YOLO-World)",
        value=objetos_default,
        help="Escriba los objetos a detectar separados por coma. Ejemplo: jeringa, guantes médicos"
    )

    imagen = st.file_uploader("Suba una imagen en formato JPG o PNG", type=["jpg", "jpeg", "png"])
    if imagen:
        img = Image.open(imagen)
        st.image(img, caption="Imagen cargada", use_container_width=True)

        modelo_yolo = YOLO("yolov8n.pt" if modelo == "YOLOv8 General" else "yolov8s-world.pt")
        if modelo == "YOLO-World" and objetos.strip():
            try:
                modelo_yolo.set_classes([o.strip().lower() for o in objetos.split(",") if o.strip()])
            except Exception as e:
                st.warning("El modelo requiere la librería CLIP para personalizar objetos.")
                st.error(str(e))
                return

        resultado = modelo_yolo(img)[0]
        st.image(resultado.plot(), caption="Resultado del análisis", use_container_width=True)

        boxes = resultado.boxes.data.cpu().numpy()
        nombres = resultado.names
        filas = []
        for box in boxes:
            x1, y1, x2, y2, score, cls = box
            nombre = nombres[int(cls)]
            traducido = LABEL_TRANSLATIONS.get(nombre.lower(), nombre)
            filas.append({
                "Objeto": traducido,
                "Confianza": f"{score * 100:.2f}%",
                "Ubicación (px)": f"[{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}]"
            })

        if filas:
            st.subheader("Objetos detectados")
            st.dataframe(pd.DataFrame(filas))
        else:
            st.info("No se detectaron objetos en la imagen proporcionada.")

def settings_section():
    st.title("Configuración")
    st.markdown("Esta sección permitirá modificar parámetros globales de la aplicación en futuras versiones.")

# --- Ruteo de herramientas según menú seleccionado ---
if st.session_state.business_type:
    if selected == "Predicción de Demanda":
        predict_demand_section()
    elif selected == "Análisis de Archivos":
        file_analysis_section()
    elif selected == "Análisis de Imágenes":
        image_analysis_section()
    elif selected == "Configuración":
        settings_section()
