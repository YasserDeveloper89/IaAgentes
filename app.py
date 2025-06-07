import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
from streamlit_option_menu import option_menu
import plotly.express as px
from ultralytics import YOLO
from PIL import Image
import io

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
        }}
        .css-1d391kg .stButton>button:hover {{
            background-color: #e67300;
            color: white;
        }}
        .stSelectbox > div {{
            font-size: 16px;
            font-weight: 600;
        }}
    </style>
""", unsafe_allow_html=True)

# --- MENÚ LATERAL ---
with st.sidebar:
    selected = option_menu(
        menu_title="Menú Principal",
        options=["Predicción Demanda", "Análisis Archivos", "Análisis de Imágenes", "Configuración"],
        icons=["bar-chart-line", "file-earmark-text", "image", "gear"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "5px", "background-color": PRIMARY_COLOR},
            "icon": {"color": "white", "font-size": "20px"},
            "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": ACCENT_COLOR},
            "nav-link-selected": {"background-color": ACCENT_COLOR},
        }
    )

# --- FUNCIONES ---

def predict_demand_section():
    st.title("📊 Predicción de Demanda")
    st.markdown("""
    Por favor, sube un archivo CSV con las siguientes columnas:

    - 📅 **fecha** (formato: YYYY-MM-DD)  
    - ☕️ **producto** (nombre del producto)  
    - 🔢 **cantidad** (unidades vendidas)
    """)
    uploaded_file = st.file_uploader("Sube tu archivo CSV aquí", type=["csv"])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file, parse_dates=['fecha'])
            df = df.sort_values(['producto', 'fecha'])
            st.subheader("Datos cargados")
            st.dataframe(df)
            
            productos = df['producto'].unique()
            producto_sel = st.selectbox("Selecciona el producto para predecir demanda", productos)
            
            df_producto = df[df['producto'] == producto_sel].copy()
            
            window = st.slider("Ventana para promedio móvil (días)", 2, 10, 3)
            growth_factor = st.slider("Factor de crecimiento esperado", 1.0, 2.0, 1.1, 0.01)
            forecast_days = st.slider("Días a predecir", 1, 14, 7)
            
            df_producto['moving_avg'] = df_producto['cantidad'].rolling(window=window).mean()
            last_avg = df_producto['moving_avg'].iloc[-1]
            
            future_dates = [df_producto['fecha'].iloc[-1] + timedelta(days=i) for i in range(1, forecast_days + 1)]
            forecast_values = [round(last_avg * (growth_factor ** i)) for i in range(1, forecast_days + 1)]
            
            forecast_df = pd.DataFrame({'fecha': future_dates, 'cantidad prevista': forecast_values})
            
            st.subheader(f"Pronóstico de demanda para: {producto_sel}")
            combined = pd.concat([
                df_producto.set_index('fecha')['cantidad'],
                forecast_df.set_index('fecha')['cantidad prevista']
            ]).reset_index()
            
            fig = px.line(combined, x='fecha', y=['cantidad', 'cantidad prevista'], 
                          labels={'value': 'Cantidad', 'fecha': 'Fecha', 'variable': 'Tipo'},
                          title="Demanda Histórica y Pronóstico")
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### Tabla de pronóstico")
            st.dataframe(forecast_df)
            
        except Exception as e:
            st.error(f"Error al procesar el archivo: {e}")
    else:
        st.info("Carga un archivo CSV para comenzar la predicción.")

def file_analysis_section():
    st.title("🔍 Análisis Exploratorio de Archivos CSV")
    st.markdown("""
    Sube un archivo CSV para visualizar sus datos, estadísticas y gráficos descriptivos.
    """)
    uploaded_file = st.file_uploader("Sube tu archivo CSV aquí", type=["csv"])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.subheader("Vista previa de datos (primeras 10 filas)")
            st.dataframe(df.head(10))
            
            st.subheader("Estadísticas descriptivas")
            st.write(df.describe(include='all'))
            
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            if numeric_cols:
                col_sel = st.selectbox("Selecciona una columna numérica para visualizar su distribución", numeric_cols)
                fig = px.histogram(df, x=col_sel, nbins=30, title=f"Histograma de {col_sel}")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No hay columnas numéricas en el archivo para generar gráficos.")
            
        except Exception as e:
            st.error(f"Error al procesar el archivo: {e}")
    else:
        st.info("Sube un archivo CSV para comenzar el análisis.")

def image_analysis_section():
    st.title("📸 Análisis Inteligente de Imágenes con IA")
    st.markdown("""
    Sube una imagen para detectar objetos usando IA avanzada (YOLOv8).
    """)
    uploaded_file = st.file_uploader("Sube imagen (jpg, png)", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        try:
            img_bytes = uploaded_file.read()
            img = Image.open(io.BytesIO(img_bytes))
            st.image(img, caption="Imagen cargada", use_column_width=True)
            
            # Carga modelo YOLOv8 preentrenado
            model = YOLO('yolov8n.pt')  # ultralytics debe estar instalado
            
            results = model(img)
            
            st.markdown("### Resultados de detección:")
            res_img = results[0].plot()
            st.image(res_img, caption="Objetos detectados", use_column_width=True)
            
            st.markdown("#### Detalles detectados:")
            detections = results[0].boxes.data.cpu().numpy() if results[0].boxes is not None else []
            if len(detections) > 0:
                labels = results[0].names
                data = []
                for box in detections:
                    x1, y1, x2, y2, score, class_id = box
                    data.append({
                        "Etiqueta": labels[int(class_id)],
                        "Confianza": round(score, 3),
                        "Coordenadas": f"({int(x1)}, {int(y1)}) - ({int(x2)}, {int(y2)})"
                    })
                st.table(data)
            else:
                st.info("No se detectaron objetos.")
                
        except Exception as e:
            st.error(f"Error al procesar la imagen: {e}")
    else:
        st.info("Sube una imagen para comenzar el análisis.")

def settings_section():
    st.title("⚙️ Configuración")
    st.markdown("""
    Aquí puedes agregar configuraciones futuras para personalizar la aplicación.
    """)

# --- RUTEO DE SECCIONES ---
if selected == "Predicción Demanda":
    predict_demand_section()
elif selected == "Análisis Archivos":
    file_analysis_section()
elif selected == "Análisis de Imágenes":
    image_analysis_section()
elif selected == "Configuración":
    settings_section()
else:
    st.write("Selecciona una opción del menú.")
