import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
from streamlit_option_menu import option_menu
import plotly.express as px
from ultralytics import YOLO
from PIL import Image
import json
import os
import cv2
import tempfile
import plotly.graph_objects as go

# --- Nuevas importaciones para dibujar zonas y stream en tiempo real ---
from streamlit_drawable_canvas import st_canvas
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, ClientSettings
import av # Necesario para streamlit-webrtc

# --- Importamos las funciones auxiliares de video_utils.py ---
from video_utils import VideoProcessor, process_video_file, load_zones, save_zones, CONFIG_ZONES_FILE

# --- Configuración de página y estilos (sin cambios) ---
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

# --- Lógica de selección de tipo de negocio (sin cambios) ---
if 'business_type' not in st.session_state:
    st.session_state.business_type = None

if st.session_state.business_type is None:
    st.title("Plataforma de Herramientas inteligentes para Restaurantes y Clínicas")
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
            menu_title="Herramientas",
            options=[
                "Predicción de demanda",
                "Análisis de archivos",
                "Análisis de imágenes",
                "Análisis de vídeo",
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

    # --- Funciones de sección (sin cambios en las primeras) ---
    def predict_demand_section():
        st.title("📈 Predicción de demanda")
        st.markdown("Suba un archivo CSV con columnas `fecha`, `elemento`, `cantidad`. Se proyectará la demanda futura de un producto.")

        archivo = st.file_uploader("Suba su archivo CSV", type=["csv"])
        if archivo:
            df = pd.read_csv(archivo, parse_dates=["fecha"])
            if not all(col in df.columns for col in ["fecha", "elemento", "cantidad"]):
                st.error("El archivo debe contener: fecha, elemento y cantidad.")
                return

            st.dataframe(df.head())

            producto = st.selectbox("Seleccione un producto", df["elemento"].unique())
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
        st.title("📂 Análisis de archivos CSV")
        st.markdown("Cargue un archivo CSV para obtener estadísticas descriptivas y gráficas automáticas.")

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
                col = st.selectbox("Columna numérica para gráficas", columnas)
                st.plotly_chart(px.histogram(df, x=col, nbins=30))
                st.plotly_chart(px.box(df, y=col))

    def image_analysis_section():
        st.title("🖼 Análisis de imágenes con IA")
        st.markdown("Suba una imagen y detecte automáticamente objetos relevantes para su negocio usando modelos de visión por computadora.")

        modelo = st.radio("Modelo de detección", ["YOLOv8 General", "YOLO-World"])

        objetos_por_defecto = (
            "strawberry, grape, banana, empanada, pizza, plate, knife, fork"
            if st.session_state.business_type == "Restaurante"
            else "face mask, syringe, medical gloves, thermometer, hospital bed"
        )

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
                    st.warning("Error con CLIP. Asegúrese de tener la librería adecuada instalada.")
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

    # --- Nueva función para la CONFIGURACIÓN DE ZONAS en la sección de Configuración ---
    def configure_zones_section():
        st.subheader("Zonas de Análisis de Vídeo")
        st.markdown("Define las áreas (mesas, zonas de espera, etc.) en tu espacio para el análisis de vídeo inteligente.")
        st.warning("Sube una imagen de referencia clara (sin personas) para delimitar las zonas. Esta configuración afectará a la sección 'Análisis de Vídeo'.")

        uploaded_img = st.file_uploader("Sube una imagen de referencia (JPG o PNG)", type=["jpg", "jpeg", "png"])
        
        drawing_mode = st.selectbox(
            "Modo de dibujo:",
            ("polygon", "rect"),
            index=0 # Default to polygon
        )

        stroke_width = st.slider("Grosor del borde", 1, 10, 2)
        stroke_color = st.color_picker("Color del borde", "#00BCD4")
        bg_color = st.color_picker("Color de relleno (transparente)", "#00BCD4") # Transparent color for fill

        canvas_result = None
        if uploaded_img is not None:
            # Display image and canvas
            img_data = uploaded_img.getvalue()
            st.image(img_data, caption="Imagen de Referencia", use_container_width=True)

            canvas_result = st_canvas(
                fill_color=bg_color + "30", # Add transparency to fill color
                stroke_width=stroke_width,
                stroke_color=stroke_color,
                background_image=Image.open(uploaded_img),
                update_streamlit=True,
                height=400,
                drawing_mode=drawing_mode,
                key="canvas",
            )

        if canvas_result and canvas_result.json_data is not None:
            objects = pd.json_normalize(canvas_result.json_data["objects"])
            
            # Allow user to name each object
            current_zones = []
            st.subheader("Nombra tus zonas:")
            for i, obj in objects.iterrows():
                if obj['type'] in ['rect', 'polygon']:
                    default_name = f"Zona {i+1}"
                    if 'label' in obj and obj['label']: # If a label was previously set
                        default_name = obj['label']

                    zone_name = st.text_input(f"Nombre para la {obj['type']} {i+1}:", value=default_name, key=f"zone_name_{i}")
                    
                    # Store coordinates based on type
                    if obj['type'] == 'rect':
                        # Rectangles have left, top, width, height
                        coords = [
                            [obj['left'], obj['top']],
                            [obj['left'] + obj['width'], obj['top']],
                            [obj['left'] + obj['width'], obj['top'] + obj['height']],
                            [obj['left'], obj['top'] + obj['height']]
                        ]
                    elif obj['type'] == 'polygon':
                        # Polygons have path
                        coords = [[p[0], p[1]] for p in obj['path']] # path is list of [x,y] points

                    current_zones.append({"name": zone_name, "coords": coords})
            
            st.session_state['current_configured_zones'] = current_zones

            if st.button("Guardar Configuración de Zonas"):
                save_zones(current_zones)
                st.success("Configuración de zonas guardada exitosamente.")
                st.rerun() # Rerun to clear canvas if desired, or just to show success

        # Display saved zones if any
        loaded_zones = load_zones()
        if loaded_zones:
            st.subheader("Zonas configuradas actualmente:")
            for zone in loaded_zones:
                st.write(f"- **{zone['name']}** (Puntos: {len(zone['coords'])})")
        else:
            st.info("No hay zonas configuradas. ¡Empieza a dibujar!")


    # --- Sección de Análisis de Vídeo (Actualizada) ---
    def video_analysis_section():
        st.title("🎥 Análisis de vídeo inteligente")
        st.markdown("Analiza la ocupación de mesas o el conteo de personas, ya sea desde un vídeo grabado o en tiempo real.")

        # Selector de modalidad: Archivo vs. Tiempo Real
        mode = st.radio(
            "Seleccione el modo de análisis de vídeo:",
            ("Cargar Archivo de Vídeo", "Stream en Tiempo Real"),
            horizontal=True
        )

        zones_configured = load_zones()
        if not zones_configured and (mode == "Cargar Archivo de Vídeo" or mode == "Stream en Tiempo Real"):
            st.warning("¡Atención! No se han configurado zonas de análisis. Por favor, vaya a la sección 'Configuración' para definirlas.")
            st.info("Sin zonas configuradas, el análisis de vídeo solo mostrará la detección global de personas (como antes).")
            # Clear people_in_zones dashboard if no zones are loaded
            if 'realtime_people_data' in st.session_state:
                del st.session_state['realtime_people_data']


        if mode == "Cargar Archivo de Vídeo":
            st.subheader("Análisis de Vídeo Grabado")
            st.markdown("Suba un vídeo corto para analizar el conteo de personas y la ocupación de mesas.")

            video_file = st.file_uploader("Seleccione un vídeo (MP4, MOV, AVI)", type=["mp4", "mov", "avi"])
            if video_file:
                # El código de procesamiento de archivo se adapta para usar las zonas
                process_video_file(video_file, zones_configured)

        elif mode == "Stream en Tiempo Real":
            st.subheader("Análisis de Vídeo en Tiempo Real (Cámara)")
            st.markdown("Conéctate a una cámara en vivo para monitorear la ocupación de mesas o el flujo de personas en tiempo real.")

            # Placeholder para mostrar el dashboard de ocupación en tiempo real
            st.session_state['dashboard_placeholder'] = st.empty()

            # Configuración de cliente para webrtc (puedes ajustar si necesitas STUN/TURN servers)
            client_settings = ClientSettings(
                rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
                media_stream_constraints={"video": True, "audio": False},
            )

            # --- Conexión al stream en tiempo real ---
            webrtc_ctx = webrtc_streamer(
                key="realtime_video_stream",
                video_processor_factory=VideoProcessor,
                client_settings=client_settings,
                async_processing=True, # Procesa frames en un hilo separado
            )
            
            # Si el stream está activo, mostrar el dashboard
            if webrtc_ctx.video_processor:
                # Update the dashboard in a loop. st.empty() is key here.
                with st.session_state['dashboard_placeholder'].container():
                    st.subheader("Ocupación de Mesas (Tiempo Real)")
                    if 'realtime_people_data' in st.session_state and st.session_state['realtime_people_data']:
                        # Asegúrate de que los datos de realtime_people_data sean consistentes
                        # Convertir a DataFrame solo si hay datos válidos
                        if isinstance(st.session_state['realtime_people_data'], dict):
                            realtime_df = pd.DataFrame(st.session_state['realtime_people_data'].items(), columns=["Zona", "Personas"])
                            realtime_df = realtime_df.sort_values(by="Personas", ascending=False)
                            
                            cols_metrics = st.columns(len(realtime_df))
                            for i, row in realtime_df.iterrows():
                                # Usar floor division para evitar IndexError si hay menos de 3 columnas
                                cols_metrics[i % 3].metric(f"{row['Zona']}", f"{row['Personas']} personas")

                            st.dataframe(realtime_df, use_container_width=True)
                        else:
                            st.info("Esperando datos válidos del stream...")
                    else:
                        st.info("Esperando datos del stream...")
            elif 'realtime_people_data' in st.session_state:
                del st.session_state['realtime_people_data'] # Clear data if stream is off

    # --- Sección de Configuración (Actualizada para incluir Configuración de Zonas) ---
    def settings_section():
        st.title("⚙️ Configuración")
        st.markdown("Aquí podrá personalizar ajustes generales de la plataforma.")

        st.subheader("Ajustes del Negocio")
        # Aquí puedes añadir cualquier otro ajuste de configuración general en el futuro.
        # Por ejemplo, umbrales, nombres por defecto, etc.

        st.markdown("---") # Separador visual
        configure_zones_section() # ¡Nueva sección para configurar las zonas!

    # --- Ruteo final de herramientas (sin cambios) ---
    if selected == "Predicción de demanda":
        predict_demand_section()
    elif selected == "Análisis de archivos":
        file_analysis_section()
    elif selected == "Análisis de imágenes":
        image_analysis_section()
    elif selected == "Análisis de vídeo":
        video_analysis_section()
    elif selected == "Configuración":
        settings_section()

