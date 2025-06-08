import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import json
import os
import tempfile
import plotly.graph_objects as go
import pandas as pd
from streamlit_webrtc import VideoProcessorBase, ClientSettings
import av

# --- ARCHIVO PARA GUARDAR Y CARGAR ZONAS ---
CONFIG_ZONES_FILE = "config_zones.json"

def load_zones():
    if os.path.exists(CONFIG_ZONES_FILE):
        try:
            with open(CONFIG_ZONES_FILE, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            st.error(f"Error al leer el archivo de configuración de zonas: {CONFIG_ZONES_FILE}. Asegúrate de que sea un JSON válido.")
            return []
    return []

def save_zones(zones):
    with open(CONFIG_ZONES_FILE, "w") as f:
        json.dump(zones, f, indent=4)

# --- Clase para procesar frames de vídeo en tiempo real ---
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        # Carga del modelo YOLO se hace una vez por instancia del procesador
        # Esto asegura que el modelo no se recargue con cada frame.
        self.model = YOLO("yolov8n.pt")
        self.zones = load_zones() # Carga las zonas configuradas
        self.zone_polygons = {} # Almacena polígonos CV2 para las zonas

        if self.zones:
            for zone in self.zones:
                # Convertir coordenadas a formato numpy para cv2.pointPolygonTest
                # Asegúrate de que las coordenadas sean enteros si vienen de `st_canvas`
                self.zone_polygons[zone['name']] = np.array(zone['coords'], np.int32).reshape((-1, 1, 2))

        # Para el conteo actual de personas en cada zona para el dashboard en tiempo real
        self.people_in_zones_current = {zone['name']: 0 for zone in self.zones}
        
        # Inicializar o actualizar el estado de la sesión para el dashboard
        if 'realtime_people_data' not in st.session_state:
            st.session_state['realtime_people_data'] = self.people_in_zones_current
        else:
            # Asegúrate de que el diccionario en session_state tenga las mismas claves de zonas
            # para evitar errores si la configuración de zonas cambia durante la ejecución
            # (aunque idealmente la app se reiniciaría al cambiar config)
            for zone_name in self.people_in_zones_current:
                if zone_name not in st.session_state['realtime_people_data']:
                    st.session_state['realtime_people_data'][zone_name] = 0


    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24") # Convertir a NumPy array (OpenCV compatible)

        # Realizar inferencia YOLO
        results = self.model(img, verbose=False)[0]
        boxes = results.boxes.data.cpu().numpy()
        
        # Reiniciar contadores para este frame
        current_frame_people_in_zones = {name: 0 for name in self.people_in_zones_current.keys()}

        # Dibujar zonas y nombres en el frame
        for zone_name, poly_coords in self.zone_polygons.items():
            cv2.polylines(img, [poly_coords], isClosed=True, color=(0, 255, 0), thickness=2)
            
            # Obtener el centro del polígono para colocar el texto del nombre de la zona
            M = cv2.moments(poly_coords)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.putText(img, zone_name, (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


        # Procesar detecciones de personas
        for box in boxes:
            x1, y1, x2, y2, conf, cls_id = box
            if int(cls_id) == 0: # Clase 'person' en YOLOv8 (cls_id 0 para persona)
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                # Dibujar bounding box de la persona
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                cv2.putText(img, f"Persona", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                # Verificar en qué zona se encuentra la persona
                point = (int(center_x), int(center_y))
                for zone_name, poly_coords in self.zone_polygons.items():
                    # cv2.pointPolygonTest devuelve >0 si dentro, 0 si en borde, <0 si fuera
                    if cv2.pointPolygonTest(poly_coords, point, False) >= 0:
                        current_frame_people_in_zones[zone_name] += 1
                        break # Asignar a la primera zona que coincida

        # Actualizar el conteo de personas y dibujarlo en el frame
        for zone_name, count in current_frame_people_in_zones.items():
            poly_coords = self.zone_polygons[zone_name]
            M = cv2.moments(poly_coords)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.putText(img, str(count), (cX - 10, cY + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Actualizar el estado de la sesión para mostrar en el dashboard de Streamlit
        st.session_state['realtime_people_data'] = current_frame_people_in_zones

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# --- Función auxiliar para procesar el archivo de vídeo subido con zonas ---
def process_video_file(video_file, zones):
    temp_file_path = None
    cap = None
    try:
        # Guardar el archivo temporalmente
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_file.write(video_file.read())
            temp_file_path = temp_file.name

        cap = cv2.VideoCapture(temp_file_path)
        model = YOLO("yolov8n.pt") # Carga tu modelo YOLO

        frame_count = 0
        data = [] # Para el conteo global de personas (como antes)
        zones_occupancy_data = {zone['name']: [] for zone in zones} # Para el conteo por zona

        zone_polygons = {}
        if zones:
            for zone in zones:
                zone_polygons[zone['name']] = np.array(zone['coords'], np.int32).reshape((-1, 1, 2))

        # Para la visualización del vídeo procesado (placeholder para el frame actual)
        output_video_placeholder = st.empty()
        
        st.write("Analizando vídeo... Esto puede tardar unos minutos.")
        progress_bar = st.progress(0)
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            st.error("No se pudo leer ningún frame del vídeo. Asegúrate de que el archivo no esté corrupto y sea compatible.")
            return

        # Limitar el número de frames a procesar para evitar tiempos de espera excesivos
        frames_to_process = min(total_frames, 300) # Procesar un máximo de 300 frames

        processed_frames_counter = 0

        with st.spinner("Procesando frames..."):
            while cap.isOpened() and processed_frames_counter < frames_to_process:
                ret, frame = cap.read()
                if not ret:
                    break

                # Opcional: Redimensionar frame para procesamiento más rápido
                # frame = cv2.resize(frame, (640, 360)) 

                results = model(frame, verbose=False)[0]
                boxes = results.boxes.data.cpu().numpy()
                
                persons_global_count = 0
                current_frame_zone_occupancy = {zone_name: 0 for zone_name in zones_occupancy_data.keys()}

                # Dibujar zonas en el frame
                for zone_name, poly_coords in zone_polygons.items():
                    cv2.polylines(frame, [poly_coords], isClosed=True, color=(0, 255, 0), thickness=2)
                    M = cv2.moments(poly_coords)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        cv2.putText(frame, zone_name, (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Procesar personas y asociarlas a zonas
                for box in boxes:
                    x1, y1, x2, y2, conf, cls_id = box
                    if int(cls_id) == 0: # Si es una persona
                        persons_global_count += 1
                        center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
                        
                        # Dibujar bounding box de la persona
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                        cv2.putText(frame, f"Persona", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                        point = (int(center_x), int(center_y))
                        for zone_name, poly_coords in zone_polygons.items():
                            if cv2.pointPolygonTest(poly_coords, point, False) >= 0:
                                current_frame_zone_occupancy[zone_name] += 1
                                break # Asignar a la primera zona que coincida
                
                # Añadir conteo global
                data.append({"Frame": frame_count, "Personas Detectadas": persons_global_count})
                
                # Añadir conteo por zona
                for zone_name, count in current_frame_zone_occupancy.items():
                    zones_occupancy_data[zone_name].append(count)
                    # Display count on the image
                    poly_coords = zone_polygons[zone_name]
                    M = cv2.moments(poly_coords)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        cv2.putText(frame, str(count), (cX - 10, cY + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                # Muestra el frame actual con las detecciones y zonas
                output_video_placeholder.image(frame, channels="BGR", use_container_width=True, caption=f"Frame {frame_count}")
                
                progress_bar.progress(min(100, int((processed_frames_counter / frames_to_process) * 100)))

                frame_count += 1
                processed_frames_counter += 1

            cap.release()
            progress_bar.empty() # Eliminar la barra de progreso una vez finalizado
            st.success("Análisis de vídeo completado.")

        # --- Visualización de resultados ---
        st.subheader("Análisis Global de Personas")
        df_global = pd.DataFrame(data)
        if not df_global.empty:
            col1, col2, col3 = st.columns(3)
            col1.metric("Frames Analizados", len(df_global))
            col2.metric("Promedio Personas", f"{df_global['Personas Detectadas'].mean():.1f}")
            col3.metric("Máximo Personas", df_global['Personas Detectadas'].max())

            fig_global = go.Figure()
            fig_global.add_trace(go.Scatter(
                x=df_global["Frame"], y=df_global["Personas Detectadas"],
                mode="lines+markers", name="Personas Globales", line=dict(color="#00bcd4")
            ))
            fig_global.update_layout(
                title="Conteo Global de Personas por Frame",
                xaxis_title="Frame", yaxis_title="Cantidad",
                plot_bgcolor="#0A0A1E", paper_bgcolor="#0A0A1E",
                font=dict(color="#E0E0E0")
            )
            st.plotly_chart(fig_global, use_container_width=True)
            st.subheader("Detalle de Datos Globales")
            st.dataframe(df_global.style.highlight_max(axis=0, color="lightgreen"), use_container_width=True)
        else:
            st.info("No se pudo realizar el análisis global de personas.")


        # --- Visualización de Ocupación por Zona ---
        if zones and any(zones_occupancy_data.values()): # Solo mostrar si hay zonas y algún dato de ocupación
            st.subheader("Análisis de Ocupación por Zonas/Mesas")
            
            # Asegurarse de que todas las listas en zones_occupancy_data tengan la misma longitud
            # para crear un DataFrame correctamente
            max_len = max(len(v) for v in zones_occupancy_data.values()) if zones_occupancy_data else 0
            for k in zones_occupancy_data:
                while len(zones_occupancy_data[k]) < max_len:
                    zones_occupancy_data[k].append(0) # Rellenar con 0 si es necesario

            df_zones_occupancy = pd.DataFrame(zones_occupancy_data)
            df_zones_occupancy['Frame'] = range(len(df_zones_occupancy))

            if not df_zones_occupancy.empty:
                fig_zones = px.line(df_zones_occupancy, x="Frame", y=list(zones_occupancy_data.keys()),
                                    title="Ocupación de Personas por Zona/Mesa a lo Largo del Tiempo")
                fig_zones.update_layout(
                    plot_bgcolor="#0A0A1E", paper_bgcolor="#0A0A1E",
                    font=dict(color="#E0E0E0"),
                    legend_title="Zona/Mesa"
                )
                st.plotly_chart(fig_zones, use_container_width=True)

                st.subheader("Resumen de Ocupación por Zona")
                summary_data = {
                    "Zona": [],
                    "Promedio de Personas": [],
                    "Máximo de Personas": []
                }
                for zone_name, counts in zones_occupancy_data.items():
                    if counts: # Asegurarse de que haya datos para la zona
                        summary_data["Zona"].append(zone_name)
                        summary_data["Promedio de Personas"].append(np.mean(counts))
                        summary_data["Máximo de Personas"].append(np.max(counts))
                
                if summary_data["Zona"]: # Verificar si hay datos para mostrar en el resumen
                    df_summary = pd.DataFrame(summary_data)
                    st.dataframe(df_summary.set_index("Zona"), use_container_width=True)
                else:
                    st.info("No se detectaron personas en ninguna zona durante el análisis del vídeo.")
            else:
                st.info("No se pudo generar el análisis de ocupación por zonas.")
        else:
            st.info("No se han configurado zonas o no se detectaron personas en las zonas configuradas.")


    finally:
        # Limpiar el archivo temporal
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        if cap:
            cap.release()

