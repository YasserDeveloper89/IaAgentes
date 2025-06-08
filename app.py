import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
from streamlit_option_menu import option_menu
import plotly.express as px
from ultralytics import YOLO
from PIL import Image

st.set_page_config(page_title="Herramientas Inteligentes", layout="wide")

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
    'jeringa': 'Jeringa', 'mascarilla': 'Mascarilla', 'guantes medicos': 'Guantes Médicos',
    'fresa': 'Fresa', 'uva': 'Uva', 'plato': 'Plato', 'vaso': 'Vaso'
}

if 'business_type' not in st.session_state:
    st.session_state.business_type = None

if st.session_state.business_type is None:
    st.title("Plataforma de Herramientas Inteligentes para Restaurantes y Clínicas")
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
            options=[
                "Predicción de Demanda",
                "Análisis de Archivos",
                "Análisis de Imágenes",
                "Análisis de Vídeo",
                "Configuración"
            ],
            icons=["bar-chart-line", "file-earmark-text", "image", "camera-video", "gear"],
            default_index=0,
            styles={
                "container": {"padding": "5px", "background-color": "#1A1A30"},
                "icon": {"color": "#00BCD4", "font-size": "20px"},
                "nav-link": {"font-size": "16px", "text-align": "left", "color": "#E0E0E0"},
                "nav-link-selected": {"background-color": "#00BCD4", "color": "#FFFFFF"}
            }
        )

    def predict_demand_section():
        st.title("📈 Predicción de Demanda")
        st.markdown("Suba un archivo CSV con las columnas: `fecha`, `elemento`, `cantidad`. El sistema proyectará la demanda futura.")

        archivo = st.file_uploader("Seleccione el archivo CSV", type=["csv"])
        if archivo:
            df = pd.read_csv(archivo, parse_dates=["fecha"])
            if not all(col in df.columns for col in ["fecha", "elemento", "cantidad"]):
                st.error("El archivo debe contener las columnas: fecha, elemento, cantidad.")
                return

            st.dataframe(df.head())

            producto = st.selectbox("Seleccione un producto para predecir", df["elemento"].unique())
            datos = df[df["elemento"] == producto].sort_values("fecha")

            ventana = st.slider("Tamaño de ventana móvil (días)", 2, 10, 3)
            crecimiento = st.slider("Crecimiento diario estimado (%)", 0, 100, 5) / 100
            dias = st.slider("Cantidad de días a predecir", 1, 30, 7)

            datos["media_movil"] = datos["cantidad"].rolling(window=ventana).mean()
            base = datos["media_movil"].dropna().iloc[-1] if not datos["media_movil"].dropna().empty else datos["cantidad"].mean()
            fechas = [datos["fecha"].max() + timedelta(days=i) for i in range(1, dias + 1)]
            cantidades = [round(base * (1 + crecimiento) ** i) for i in range(1, dias + 1)]

            pred = pd.DataFrame({"Fecha": fechas, "Cantidad Prevista": cantidades})
            st.plotly_chart(px.line(pred, x="Fecha", y="Cantidad Prevista", title=f"Proyección de demanda: {producto}"))
            st.dataframe(pred)

    def file_analysis_section():
        st.title("📂 Análisis de Archivos CSV")
        st.markdown("Explore y analice sus archivos de datos. Esta herramienta genera estadísticas y visualizaciones automáticamente.")

        archivo = st.file_uploader("Suba su archivo CSV", type=["csv"])
        if archivo:
            df = pd.read_csv(archivo)
            st.subheader("Vista previa")
            st.dataframe(df.head(10))

            st.subheader("Estadísticas generales")
            desc = df.describe(include='all').T
            desc.rename(columns={
                "count": "Cantidad", "unique": "Valores Únicos", "top": "Más Frecuente", "freq": "Frecuencia",
                "mean": "Promedio", "std": "Desviación", "min": "Mínimo", "25%": "P25", "50%": "Mediana", "75%": "P75", "max": "Máximo"
            }, inplace=True)
            st.dataframe(desc)

            columnas = df.select_dtypes(include=np.number).columns.tolist()
            if columnas:
                col = st.selectbox("Columna numérica para visualizar", columnas)
                st.plotly_chart(px.histogram(df, x=col, nbins=30))
                st.plotly_chart(px.box(df, y=col))
                def image_analysis_section():
        st.title("🖼 Análisis de Imágenes con IA")
        st.markdown("Suba una imagen y elija un modelo de detección. Puede definir objetos personalizados (en inglés) para mayor precisión.")

        modelo = st.radio("Modelo de detección", ["YOLOv8 General", "YOLO-World"])
        objetos_por_defecto = "strawberry, grape, banana, empanada, pizza, plate, knife, fork" if st.session_state.business_type == "Restaurante" else "face mask, syringe, medical gloves, thermometer, hospital bed"
        objetos = st.text_input("Objetos personalizados (solo YOLO-World)", value=objetos_por_defecto)

        archivo = st.file_uploader("Cargue una imagen (JPG o PNG)", type=["jpg", "jpeg", "png"])
        if archivo:
            imagen = Image.open(archivo)
            st.image(imagen, caption="Imagen original", use_container_width=True)
            modelo_yolo = YOLO("yolov8n.pt" if modelo == "YOLOv8 General" else "yolov8s-world.pt")

            if modelo == "YOLO-World" and objetos.strip():
                try:
                    modelo_yolo.set_classes([o.strip().lower() for o in objetos.split(",") if o.strip()])
                except Exception as e:
                    st.warning("Error al cargar objetos personalizados. ¿Está instalada la librería 'open-clip'?")
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

    def video_analysis_section():
        st.title("🎥 Análisis de Vídeo en Frames")
        st.markdown("Suba un vídeo corto. La inteligencia artificial detectará cuántas personas aparecen por cuadro.")

        video_file = st.file_uploader("Seleccione un vídeo (MP4, MOV, AVI)", type=["mp4", "mov", "avi"])
        if video_file:
            import tempfile
            import cv2
            import plotly.graph_objects as go

            temp = tempfile.NamedTemporaryFile(delete=False)
            temp.write(video_file.read())
            cap = cv2.VideoCapture(temp.name)

            model = YOLO("yolov8n.pt")
            frame_count = 0
            data = []

            with st.spinner("Procesando vídeo, espere un momento..."):
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret or frame_count > 150:
                        break
                    result = model(frame, verbose=False)[0]
                    boxes = result.boxes
                    persons = sum(1 for i in range(len(boxes.cls)) if int(boxes.cls[i]) == 0)
                    data.append({"Frame": frame_count, "Personas Detectadas": persons})
                    frame_count += 1
                cap.release()

            df = pd.DataFrame(data)

            col1, col2, col3 = st.columns(3)
            col1.metric("Frames Analizados", len(df))
            col2.metric("Promedio", f"{df['Personas Detectadas'].mean():.1f}")
            col3.metric("Máximo", df['Personas Detectadas'].max())

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df["Frame"], y=df["Personas Detectadas"],
                mode="lines+markers", name="Personas", line=dict(color="#00bcd4")
            ))
            fig.add_hline(
                y=10, line_dash="dash", line_color="red",
                annotation_text="Aforo máximo", annotation_position="top left"
            )
            fig.update_layout(
                title="Conteo de personas por cuadro",
                xaxis_title="Frame", yaxis_title="Cantidad",
                plot_bgcolor="#0A0A1E", paper_bgcolor="#0A0A1E",
                font=dict(color="#E0E0E0")
            )

            st.plotly_chart(fig, use_container_width=True)
            st.subheader("Detalle de Datos")
            st.dataframe(df.style.highlight_max(axis=0, color="lightgreen"), use_container_width=True)

    def settings_section():
        st.title("⚙️ Configuración")
        st.markdown("Espacio reservado para ajustes avanzados y personalización futura del sistema.")

    # Ruteo final de navegación
    if selected == "Predicción de Demanda":
        predict_demand_section()
    elif selected == "Análisis de Archivos":
        file_analysis_section()
    elif selected == "Análisis de Imágenes":
        image_analysis_section()
    elif selected == "Análisis de Vídeo":
        video_analysis_section()
    elif selected == "Configuración":
        settings_section()
