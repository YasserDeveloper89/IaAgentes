
import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
from streamlit_option_menu import option_menu
import plotly.express as px
from ultralytics import YOLO
from PIL import Image

# Configuración de página
st.set_page_config(page_title="Herramientas Inteligentes", layout="wide")

# Estilos personalizados
st.markdown("""
    <style>
        .stApp { background-color: #0A0A1E; color: #E0E0E0; font-family: 'Segoe UI', sans-serif; }
        .stSidebar { background-color: #1A1A30; }
        h1, h2, h3 { color: #00BCD4; }
        div.stButton > button {
            background-color: #00BCD4;
            color: white;
            font-weight: bold;
            border-radius: 5px;
            padding: 0.5em 2em;
        }
        div.stButton > button:hover {
            background-color: #009bb3;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

LABEL_TRANSLATIONS = {
    'person': 'Persona', 'bottle': 'Botella', 'cup': 'Taza',
    'syringe': 'Jeringa', 'mask': 'Mascarilla', 'gloves': 'Guantes Médicos',
    'strawberry': 'Fresa', 'grape': 'Uva', 'dish': 'Plato', 'glass': 'Vaso'
}

# Lógica de selección del negocio
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
    st.sidebar.title(f"Negocio: {st.session_state.business_type}")
    if st.sidebar.button("Cambiar tipo de negocio"):
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
                "icon": {"color": "#00BCD4", "font-size": "20px"},
                "nav-link": {"font-size": "16px", "text-align": "left", "color": "#E0E0E0"},
                "nav-link-selected": {"background-color": "#00BCD4", "color": "#FFFFFF"}
            }
        )

    # === Definición de herramientas ===

    def predict_demand_section():
        st.title("Predicción de Demanda")
        archivo = st.file_uploader("Suba un archivo CSV con columnas: fecha, elemento, cantidad", type=["csv"])
        if archivo:
            df = pd.read_csv(archivo, parse_dates=["fecha"])
            if not all(col in df.columns for col in ["fecha", "elemento", "cantidad"]):
                st.error("El archivo debe contener: fecha, elemento y cantidad.")
                return

            st.dataframe(df)
            producto = st.selectbox("Seleccione un producto", df["elemento"].unique())
            datos = df[df["elemento"] == producto].sort_values("fecha")

            ventana = st.slider("Tamaño de ventana móvil (días)", 2, 10, 3)
            crecimiento = st.slider("Crecimiento diario estimado (%)", 0, 100, 5) / 100
            dias = st.slider("Cantidad de días a predecir", 1, 30, 7)

            datos["media_movil"] = datos["cantidad"].rolling(window=ventana).mean()
            base = datos["media_movil"].dropna().iloc[-1] if not datos["media_movil"].dropna().empty else datos["cantidad"].mean()
            fechas = [datos["fecha"].max() + timedelta(days=i) for i in range(1, dias+1)]
            cantidades = [round(base * (1 + crecimiento) ** i) for i in range(1, dias+1)]
            pred = pd.DataFrame({"Fecha": fechas, "Cantidad Prevista": cantidades})

            st.plotly_chart(px.line(pred, x="Fecha", y="Cantidad Prevista", title=f"Proyección de demanda: {producto}"))
            st.dataframe(pred)

    def file_analysis_section():
        st.title("Análisis de Archivos CSV")
        archivo = st.file_uploader("Suba su archivo CSV", type=["csv"])
        if archivo:
            df = pd.read_csv(archivo)
            st.subheader("Vista previa")
            st.dataframe(df.head(10))

            st.subheader("Estadísticas")
            desc = df.describe(include='all').T
            desc.rename(columns={
                "count": "Cantidad", "unique": "Valores Únicos", "top": "Más Frecuente", "freq": "Frecuencia",
                "mean": "Promedio", "std": "Desviación", "min": "Mínimo", "25%": "P25", "50%": "Mediana", "75%": "P75", "max": "Máximo"
            }, inplace=True)
            st.dataframe(desc)

            columnas = df.select_dtypes(include=np.number).columns.tolist()
            if columnas:
                col = st.selectbox("Columna numérica", columnas)
                st.plotly_chart(px.histogram(df, x=col, nbins=30))
                st.plotly_chart(px.box(df, y=col))

    def image_analysis_section():
        st.title("Análisis de Imágenes")
        modelo = st.radio("Modelo de detección", ["YOLOv8 General", "YOLO-World"])

        objetos_por_defecto = "strawberry, grape, banana, empanada, pizza, plate, knife, fork" if st.session_state.business_type == "Restaurante" else "face mask, syringe, medical gloves, thermometer, hospital bed"
        objetos = st.text_input("Objetos personalizados (solo YOLO-World)", value=objetos_por_defecto)

        archivo = st.file_uploader("Cargue una imagen (jpg/png)", type=["jpg", "jpeg", "png"])
        if archivo:
            imagen = Image.open(archivo)
            st.image(imagen, caption="Imagen original", use_container_width=True)
            modelo_yolo = YOLO("yolov8n.pt" if modelo == "YOLOv8 General" else "yolov8s-world.pt")

            if modelo == "YOLO-World" and objetos.strip():
                try:
                    modelo_yolo.set_classes([o.strip().lower() for o in objetos.split(",") if o.strip()])
                except Exception as e:
                    st.warning("Requiere 'open-clip'.")
                    st.error(str(e))
                    return

            resultado = modelo_yolo(imagen)[0]
            st.image(resultado.plot(), caption="Resultado del análisis", use_container_width=True)

            cajas = resultado.boxes.data.cpu().numpy()
            nombres = resultado.names
            filas = []
            for box in cajas:
                x1, y1, x2, y2, conf, clase = box
                etiqueta = nombres[int(clase)]
                traduccion = LABEL_TRANSLATIONS.get(etiqueta.lower(), etiqueta)
                filas.append({
                    "Objeto Detectado": traduccion,
                    "Confianza": f"{conf*100:.2f}%",
                    "Ubicación": f"[{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}]"
                })

            if filas:
                st.subheader("Objetos detectados")
                st.dataframe(pd.DataFrame(filas))
            else:
                st.info("No se detectaron objetos en la imagen.")

    def settings_section():
        st.title("Configuración")
        st.markdown("Espacio reservado para ajustes futuros.")

    # Ruteo corregido
    if selected == "Predicción de Demanda":
        predict_demand_section()
    elif selected == "Análisis de Archivos":
        file_analysis_section()
    elif selected == "Análisis de Imágenes":
        image_analysis_section()
    elif selected == "Configuración":
        settings_section()
