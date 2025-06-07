import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
from streamlit_option_menu import option_menu
import plotly.express as px
from PIL import Image
from ultralytics import YOLO

# --- CONFIGURACIÓN VISUAL ---
PRIMARY_COLOR = "#1f77b4"
ACCENT_COLOR = "#ff7f0e"
BACKGROUND_COLOR = "#f0f2f6"
FONT_FAMILY = "Arial, sans-serif"

st.set_page_config(page_title="AI Agents + Video Analytics", layout="wide", page_icon="🤖")

# --- ESTILOS CSS PARA ESTÉTICA MODERNA ---
st.markdown(f"""
    <style>
        .css-1d391kg {{
            background-color: {BACKGROUND_COLOR};
            font-family: {FONT_FAMILY};
        }}
        .stApp {{
            background: linear-gradient(135deg, #e0eafc 0%, #cfdef3 100%);
            color: #333;
            font-family: {FONT_FAMILY};
        }}
        .sidebar .sidebar-content {{
            background-color: {PRIMARY_COLOR};
            color: white;
            font-family: {FONT_FAMILY};
        }}
        .css-1d391kg .stButton>button {{
            background-color: {ACCENT_COLOR};
            color: white;
            border-radius: 8px;
            padding: 8px 16px;
            font-weight: bold;
            transition: background-color 0.3s ease;
        }}
        .css-1d391kg .stButton>button:hover {{
            background-color: #e67300;
            color: white;
        }}
        /* Tabla estilos */
        .dataframe tbody tr:nth-child(even) {{
            background-color: #f0f2f6;
        }}
        .dataframe thead {{
            background-color: {PRIMARY_COLOR};
            color: white;
        }}
    </style>
""", unsafe_allow_html=True)

# --- MENU LATERAL ---
with st.sidebar:
    selected = option_menu(
        menu_title="Menú Principal",
        options=["Predicción Demanda", "Análisis de Archivos", "Análisis Inteligente de Imágenes", "Configuración"],
        icons=["bar-chart-line", "file-earmark-text", "image", "gear"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "5px", "background-color": PRIMARY_COLOR},
            "icon": {"color": "white", "font-size": "20px"},
            "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px", "--hover-color": ACCENT_COLOR},
            "nav-link-selected": {"background-color": ACCENT_COLOR},
        }
    )

# --- FUNCIONES DE SECCIÓN ---

def predict_demand_section():
    st.title("📊 Predicción de Demanda")
    st.markdown("""
    Carga un archivo CSV con las columnas obligatorias:  
    - **fecha** (formato YYYY-MM-DD)  
    - **producto**  
    - **cantidad** (vendida, números enteros)  
    """)
    uploaded_file = st.file_uploader("Sube archivo CSV", type=["csv"])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file, parse_dates=['fecha'])
            df = df.sort_values(['producto', 'fecha'])
            st.subheader("Datos Cargados")
            st.dataframe(df)

            productos = df['producto'].unique()
            producto_sel = st.selectbox("Selecciona producto para pronóstico", productos)

            df_producto = df[df['producto'] == producto_sel].copy()

            window = st.slider("Ventana de promedio móvil (días)", 2, 10, 3)
            growth_factor = st.slider("Factor de crecimiento esperado", 1.0, 2.0, 1.1, 0.01)
            forecast_days = st.slider("Días a predecir", 1, 14, 7)

            df_producto['moving_avg'] = df_producto['cantidad'].rolling(window=window).mean()
            last_avg = df_producto['moving_avg'].iloc[-1]

            future_dates = [df_producto['fecha'].iloc[-1] + timedelta(days=i) for i in range(1, forecast_days + 1)]
            forecast_values = [int(round(last_avg * (growth_factor ** i))) for i in range(1, forecast_days + 1)]

            forecast_df = pd.DataFrame({'fecha': future_dates, 'Cantidad Pronosticada': forecast_values})

            st.subheader(f"Pronóstico para: {producto_sel}")
            combined = pd.concat([
                df_producto.set_index('fecha')['cantidad'],
                forecast_df.set_index('fecha')['Cantidad Pronosticada']
            ])
            fig = px.line(combined, labels={'index': 'Fecha', 'value': 'Cantidad'}, title="Demanda Histórica y Pronóstico")
            st.plotly_chart(fig, use_container_width=True)
            st.write(forecast_df)

        except Exception as e:
            st.error(f"Error al procesar el archivo: {e}")
    else:
        st.info("Carga un archivo CSV para empezar.")

def file_analysis_section():
    st.title("🔍 Análisis de Archivos")
    st.markdown("""
    Carga un archivo CSV para explorarlo y visualizar estadísticas básicas de manera clara y amigable.
    """)
    uploaded_file = st.file_uploader("Sube archivo CSV", type=["csv"])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.subheader("Vista previa de datos (primeras 10 filas)")
            st.dataframe(df.head(10))

            st.subheader("Descripción estadística resumida")
            st.write(df.describe(include='all').T)

            st.subheader("Visualización rápida")
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            if numeric_cols:
                col_sel = st.selectbox("Selecciona una columna numérica para graficar", numeric_cols)
                fig = px.histogram(df, x=col_sel, nbins=30, title=f"Histograma de '{col_sel}'")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No se encontraron columnas numéricas para graficar.")
        except Exception as e:
            st.error(f"Error procesando el archivo: {e}")
    else:
        st.info("Sube un archivo CSV para análisis.")

def image_analysis_section():
    st.title("🖼️ Análisis Inteligente de Imágenes")
    st.markdown("""
    Sube una imagen para detectar objetos y personas automáticamente.  
    La aplicación mostrará la imagen con los cuadros de detección y un resumen claro de los objetos detectados.
    """)

    uploaded_image = st.file_uploader("Carga una imagen (jpg, png)", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        try:
            image = Image.open(uploaded_image).convert("RGB")
            st.image(image, caption="Imagen cargada", use_container_width=True)

            # Cargar modelo YOLO preentrenado
            model = YOLO("yolov8n.pt")
            results = model(image)

            # Mostrar imagen con detecciones
            annotated_frame = results[0].plot()
            st.image(annotated_frame, caption="Imagen con detecciones", use_container_width=True)

            # Procesar resultados para tabla
            detections = results[0].boxes.data.cpu().numpy() if results[0].boxes.data is not None else []
            class_ids = results[0].boxes.cls.cpu().numpy() if results[0].boxes.cls is not None else []
            confidences = results[0].boxes.conf.cpu().numpy() if results[0].boxes.conf is not None else []
            names = results[0].names

            if len(detections) > 0:
                datos = []
                for i, box in enumerate(detections):
                    clase = names[int(class_ids[i])]
                    confianza = confidences[i]
                    datos.append({
                        "Objeto detectado": clase.capitalize(),
                        "Confianza (%)": f"{confianza*100:.1f}",
                        "Coordenadas (xmin, ymin, xmax, ymax)": f"{int(box[0])}, {int(box[1])}, {int(box[2])}, {int(box[3])}"
                    })
                df = pd.DataFrame(datos)

                st.markdown("### 📋 Resumen de objetos detectados")
                st.dataframe(df, use_container_width=True)

                # Mostrar resumen con estilo
                total_objetos = len(detections)
                st.markdown(f"**Total de objetos detectados:** {total_objetos}")

                # Conteo por tipo de objeto
                conteo = df['Objeto detectado'].value_counts().reset_index()
                conteo.columns = ["Objeto", "Cantidad"]
                st.markdown("### 📊 Conteo por tipo de objeto")
                st.table(conteo.style.set_table_styles([
                    {'selector': 'th', 'props': [('background-color', PRIMARY_COLOR), ('color', 'white'), ('font-weight', 'bold')]},
                    {'selector': 'td', 'props': [('padding', '8px')]},
                    {'selector': 'tr:nth-child(even)', 'props': [('background-color', '#f0f2f6')]}
                ]))

            else:
                st.info("No se detectaron objetos
