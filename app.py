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
        /* Estilos generales para el fondo y la fuente de la aplicación */
        .stApp {{
            background: linear-gradient(135deg, #e0eafc 0%, #cfdef3 100%);
            color: #333;
            font-family: {FONT_FAMILY};
        }}
        /* Estilos para el sidebar */
        .sidebar .sidebar-content {{
            background-color: {PRIMARY_COLOR};
            color: white;
            font-family: {FONT_FAMILY};
        }}
        /* Estilos para los botones de Streamlit */
        .stButton>button {{
            background-color: {ACCENT_COLOR};
            color: white;
            border-radius: 8px;
            padding: 8px 16px;
            font-weight: bold;
            border: none; /* Eliminar borde para un look más limpio */
            transition: background-color 0.3s ease; /* Suavizar transición al hover */
        }}
        .stButton>button:hover {{
            background-color: #e67300;
            color: white;
        }}
        /* Estilos para los selectbox y otros widgets */
        .stSelectbox > div, .stSlider > div > div > div, .stRadio > label {{
            font-size: 16px;
            font-weight: 600;
            color: #333;
        }}
        /* Mejorar la legibilidad de los títulos de sección */
        h1 {{
            color: {PRIMARY_COLOR};
            text-align: center;
            font-weight: 700;
            margin-bottom: 20px;
        }}
        h3 {{
            color: {PRIMARY_COLOR};
            font-weight: 600;
            margin-top: 25px;
            margin-bottom: 15px;
        }}
        h4 {{
            color: #555;
            font-weight: 500;
            margin-top: 20px;
            margin-bottom: 10px;
        }}
        /* Estilos para los mensajes de información/error/éxito */
        .stAlert {{
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 20px;
        }}
        /* Estilo para tablas de Streamlit para mejor visualización */
        .dataframe {{
            font-size: 0.9em;
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 20px;
        }}
        .dataframe th, .dataframe td {{
            padding: 8px 12px;
            border: 1px solid #ddd;
            text-align: left;
        }}
        .dataframe th {{
            background-color: {PRIMARY_COLOR};
            color: white;
            font-weight: bold;
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

# Diccionario para traducir las etiquetas de YOLOv8 a español
# Puedes añadir más traducciones aquí según las clases que detecte tu modelo
LABEL_TRANSLATIONS = {
    'person': 'Persona',
    'bicycle': 'Bicicleta',
    'car': 'Coche',
    'motorcycle': 'Motocicleta',
    'airplane': 'Avión',
    'bus': 'Autobús',
    'train': 'Tren',
    'truck': 'Camión',
    'boat': 'Barco',
    'traffic light': 'Semáforo',
    'fire hydrant': 'Boca de Incendios',
    'stop sign': 'Señal de Stop',
    'parking meter': 'Parquímetro',
    'bench': 'Banco',
    'bird': 'Pájaro',
    'cat': 'Gato',
    'dog': 'Perro',
    'horse': 'Caballo',
    'sheep': 'Oveja',
    'cow': 'Vaca',
    'elephant': 'Elefante',
    'bear': 'Oso',
    'zebra': 'Cebra',
    'giraffe': 'Jirafa',
    'backpack': 'Mochila',
    'umbrella': 'Paraguas',
    'handbag': 'Bolso',
    'tie': 'Corbata',
    'suitcase': 'Maleta',
    'frisbee': 'Frisbee',
    'skis': 'Esquís',
    'snowboard': 'Tabla de Snow',
    'sports ball': 'Pelota de Deporte',
    'kite': 'Cometa',
    'baseball bat': 'Bate de Béisbol',
    'baseball glove': 'Guante de Béisbol',
    'skateboard': 'Monopatín',
    'surfboard': 'Tabla de Surf',
    'tennis racket': 'Raqueta de Tenis',
    'bottle': 'Botella',
    'wine glass': 'Copa de Vino',
    'cup': 'Taza',
    'fork': 'Tenedor',
    'knife': 'Cuchillo',
    'spoon': 'Cuchara',
    'bowl': 'Cuenco',
    'banana': 'Plátano',
    'apple': 'Manzana',
    'sandwich': 'Sándwich',
    'orange': 'Naranja',
    'broccoli': 'Brócoli',
    'carrot': 'Zanahoria',
    'hot dog': 'Perrito Caliente',
    'pizza': 'Pizza',
    'donut': 'Dona',
    'cake': 'Tarta',
    'chair': 'Silla',
    'couch': 'Sofá',
    'potted plant': 'Planta en Maceta',
    'bed': 'Cama',
    'dining table': 'Mesa de Comedor',
    'toilet': 'Inodoro',
    'tv': 'Televisión',
    'laptop': 'Portátil',
    'mouse': 'Ratón',
    'remote': 'Mando a Distancia',
    'keyboard': 'Teclado',
    'cell phone': 'Teléfono Móvil',
    'microwave': 'Microondas',
    'oven': 'Horno',
    'toaster': 'Tostadora',
    'sink': 'Fregadero',
    'refrigerator': 'Refrigerador',
    'book': 'Libro',
    'clock': 'Reloj',
    'vase': 'Jarrón',
    'scissors': 'Tijeras',
    'teddy bear': 'Oso de Peluche',
    'hair drier': 'Secador de Pelo',
    'toothbrush': 'Cepillo de Dientes'
}


def predict_demand_section():
    st.title("📊 Predicción de Demanda")
    st.markdown("""
    Esta herramienta te permite pronosticar la demanda futura de tus productos basándose en datos históricos.
    Por favor, sube un archivo CSV con las siguientes columnas:

    - 📅 **fecha** (formato:YYYY-MM-DD)  
    - ☕️ **producto** (nombre del producto, por ejemplo: "Café", "Té", "Pastel")  
    - 🔢 **cantidad** (unidades vendidas o demandadas)

    **Instrucciones:**
    1. Sube tu archivo CSV utilizando el botón de abajo.
    2. Selecciona el producto para el cual deseas generar la predicción.
    3. Ajusta la "Ventana para promedio móvil" para suavizar los datos históricos.
    4. Modifica el "Factor de crecimiento esperado" según tus expectativas futuras.
    5. Define el número de "Días a predecir" para tu pronóstico.
    """)

    uploaded_file = st.file_uploader("Sube tu archivo CSV de datos históricos aquí", type=["csv"])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file, parse_dates=['fecha'])
            df = df.sort_values(['producto', 'fecha'])

            st.subheader("✅ Datos históricos cargados correctamente")
            st.dataframe(df) # Muestra todo el DataFrame cargado inicialmente

            # Filtrar datos por producto
            productos = df['producto'].unique()
            producto_sel = st.selectbox("👉 Selecciona el producto para el cual deseas predecir la demanda", productos)

            df_producto = df[df['producto'] == producto_sel].copy()

            # Parámetros para la predicción
            st.subheader("⚙️ Parámetros de Predicción")
            window = st.slider("Ventana para promedio móvil (número de días a considerar para el promedio)", 2, 10, 3, help="Define cuántos días anteriores se usan para calcular el promedio móvil. Un valor más alto suaviza las fluctuaciones.")
            growth_factor = st.slider("Factor de crecimiento esperado (ej: 1.1 para un 10% de crecimiento diario)", 1.0, 2.0, 1.05, 0.01, help="Establece el factor por el cual se espera que la demanda crezca cada día en el futuro. 1.0 significa sin crecimiento, 1.1 un 10% de crecimiento diario.")
            forecast_days = st.slider("Días a predecir (número de días futuros para el pronóstico)", 1, 30, 7, help="Determina cuántos días hacia adelante deseas pronosticar la demanda.")

            # Cálculo de promedio móvil y predicción
            df_producto['promedio_movil'] = df_producto['cantidad'].rolling(window=window).mean().round(2) # Redondeado para mejor visualización
            
            # Asegurarse de que hay datos suficientes para el promedio
            if not df_producto['promedio_movil'].dropna().empty:
                last_avg = df_producto['promedio_movil'].iloc[-1]
            else:
                st.warning("No hay suficientes datos para calcular el promedio móvil. Ajusta la ventana o el dataset.")
                last_avg = df_producto['cantidad'].mean() if not df_producto['cantidad'].empty else 0
                st.info(f"Usando el promedio simple de cantidad ({last_avg:.2f}) para la predicción debido a datos insuficientes para el promedio móvil.")


            future_dates = [df_producto['fecha'].iloc[-1] + timedelta(days=i) for i in range(1, forecast_days + 1)]
            # Calcular el pronóstico aplicando el factor de crecimiento acumulativamente
            forecast_values = []
            current_forecast_value = last_avg
            for i in range(forecast_days):
                current_forecast_value *= growth_factor
                forecast_values.append(round(current_forecast_value)) # Redondeado a entero para cantidades

            forecast_df = pd.DataFrame({'fecha': future_dates, 'cantidad_prevista': forecast_values})

            st.subheader(f"📈 Pronóstico de Demanda para: **{producto_sel}**")

            # Combinar datos históricos y previstos para el gráfico
            combined = pd.concat([
                df_producto.set_index('fecha')['cantidad'].rename('Cantidad Histórica'),
                forecast_df.set_index('fecha')['cantidad_prevista'].rename('Cantidad Prevista')
            ], axis=1).reset_index()
            
            # Gráfico interactivo con Plotly Express
            fig = px.line(
                combined, 
                x='fecha', 
                y=['Cantidad Histórica', 'Cantidad Prevista'],
                title=f'Demanda Histórica y Pronóstico para {producto_sel}',
                labels={'value': 'Cantidad', 'fecha': 'Fecha', 'variable': 'Tipo de Cantidad'},
                color_discrete_map={
                    'Cantidad Histórica': PRIMARY_COLOR, 
                    'Cantidad Prevista': ACCENT_COLOR
                },
                template="plotly_white" # Estilo de fondo del gráfico
            )
            
            # Mejoras al gráfico: líneas más gruesas, marcadores, rango en el eje Y
            fig.update_traces(mode='lines+markers', hovertemplate="Fecha: %{x}<br>Cantidad: %{y}<extra></extra>")
            fig.update_layout(
                hovermode="x unified",
                xaxis_title="Fecha",
                yaxis_title="Cantidad de Unidades",
                legend_title="Leyenda",
                font=dict(family=FONT_FAMILY),
                margin=dict(l=0, r=0, t=50, b=0) # Ajustar márgenes
            )
            # Añadir una línea vertical para separar datos históricos de previstos
            if not df_producto.empty:
                last_historical_date = df_producto['fecha'].iloc[-1]
                fig.add_vline(x=last_historical_date, line_width=2, line_dash="dash", line_color="grey", 
                              annotation_text="Inicio del Pronóstico", annotation_position="top right")

            st.plotly_chart(fig, use_container_width=True) # <-- CORRECCIÓN: use_container_width
            
            st.markdown("### 📋 Tabla Detallada del Pronóstico")
            # Presenta la tabla de pronóstico con un formato más amigable
            st.dataframe(forecast_df.rename(columns={'fecha': 'Fecha', 'cantidad_prevista': 'Cantidad Prevista (Unidades)'}).set_index('Fecha'))
            
        except pd.errors.EmptyDataError:
            st.error("El archivo CSV está vacío. Por favor, sube un archivo con datos.")
        except KeyError as ke:
            st.error(f"Faltan columnas requeridas en tu CSV. Asegúrate de tener 'fecha', 'producto' y 'cantidad'. Error: {ke}")
        except Exception as e:
            st.error(f"Ocurrió un error inesperado al procesar el archivo: {e}")
            st.info("Asegúrate de que el formato de la fecha sea YYYY-MM-DD y que todas las columnas necesarias estén presentes.")
    else:
        st.info("⬆️ Carga tu archivo CSV para comenzar el análisis de predicción de demanda.")


def file_analysis_section():
    st.title("🔍 Análisis Exploratorio de Archivos CSV")
    st.markdown("""
    Esta sección te permite explorar y entender la estructura de tus archivos CSV.
    Sube un archivo CSV para visualizar sus datos, estadísticas descriptivas y la distribución de sus columnas numéricas.
    """)
    uploaded_file = st.file_uploader("Sube tu archivo CSV aquí", type=["csv"])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            
            st.subheader("✅ Vista Previa de los Datos (Primeras 10 filas)")
            st.dataframe(df.head(10))
            
            st.subheader("📊 Estadísticas Descriptivas")
            st.write(df.describe(include='all').T.round(2)) # .T para transponer y round para mejor lectura
            
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            if numeric_cols:
                st.subheader("📈 Visualización de Distribuciones Numéricas")
                col_sel = st.selectbox("👉 Selecciona una columna numérica para visualizar su distribución (Histograma)", numeric_cols)
                
                fig = px.histogram(df, x=col_sel, nbins=30, 
                                   title=f"Distribución de Frecuencia de '{col_sel}'",
                                   labels={col_sel: col_sel, 'count': 'Frecuencia'},
                                   template="plotly_white",
                                   color_discrete_sequence=[PRIMARY_COLOR]) # Color del histograma
                
                fig.update_layout(
                    xaxis_title=col_sel,
                    yaxis_title="Frecuencia de Aparición",
                    font=dict(family=FONT_FAMILY),
                    margin=dict(l=0, r=0, t=50, b=0)
                )
                st.plotly_chart(fig, use_container_width=True) # <-- CORRECCIÓN: use_container_width

                # Gráfico de caja para outliers
                st.subheader(f"📦 Detección de Valores Atípicos para '{col_sel}'")
                fig_box = px.box(df, y=col_sel, 
                                 title=f"Diagrama de Caja de '{col_sel}' (Identificación de Atípicos)",
                                 labels={col_sel: col_sel},
                                 template="plotly_white",
                                 color_discrete_sequence=[ACCENT_COLOR])
                fig_box.update_layout(
                    yaxis_title=col_sel,
                    font=dict(family=FONT_FAMILY),
                    margin=dict(l=0, r=0, t=50, b=0)
                )
                st.plotly_chart(fig_box, use_container_width=True) # <-- CORRECCIÓN: use_container_width

            else:
                st.info("ℹ️ No se encontraron columnas numéricas en el archivo para generar gráficos de distribución.")
            
        except pd.errors.EmptyDataError:
            st.error("El archivo CSV está vacío. Por favor, sube un archivo con datos.")
        except Exception as e:
            st.error(f"Ocurrió un error inesperado al procesar el archivo: {e}")
            st.info("Asegúrate de que el archivo es un CSV válido y no está corrupto.")
    else:
        st.info("⬆️ Sube un archivo CSV para comenzar el análisis exploratorio.")

def image_analysis_section():
    st.title("📸 Análisis Inteligente de Imágenes con IA")
    st.markdown("""
    Esta sección utiliza modelos avanzados de Inteligencia Artificial (YOLOv8) para detectar y clasificar objetos dentro de las imágenes que subas.
    Simplemente carga tu imagen y la IA hará el resto.
    """)
    uploaded_file = st.file_uploader("Sube tu imagen aquí (formatos soportados: JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        try:
            st.info("✅ Imagen subida. Procesando...") # Mensaje de depuración
            img_bytes = uploaded_file.read()
            img = Image.open(io.BytesIO(img_bytes))
            st.image(img, caption="Imagen Original Cargada", use_container_width=True) # <-- CORRECCIÓN: use_container_width
            
            st.info("⚙️ Cargando el modelo de detección de objetos (YOLOv8n)...") # Mensaje de depuración
            model = YOLO('yolov8n.pt')  # El modelo se descargará si no está en caché
            st.info("✅ Modelo YOLOv8 cargado. Iniciando detección...") # Mensaje de depuración
            
            # Realiza la detección de objetos
            results = model(img)
            st.info("✨ Detección de objetos completada.") # Mensaje de depuración
            
            st.subheader("🖼️ Imagen con Objetos Detectados")
            # results[0].plot() devuelve una imagen de NumPy con las cajas y etiquetas dibujadas
            res_img_array = results[0].plot() # Esto es un array de NumPy
            st.image(res_img_array, caption="Objetos Detectados por la IA", use_container_width=True) # <-- CORRECCIÓN: use_container_width
            
            st.markdown("### 📋 Detalles de los Objetos Detectados")
            detections = results[0].boxes.data.cpu().numpy() if results[0].boxes is not None else []
            
            if len(detections) > 0:
                labels = results[0].names
                data = []
                for box in detections:
                    # Coordenadas de la caja y puntuación de confianza
                    x1, y1, x2, y2, score, class_id = box
                    original_label = labels[int(class_id)]
                    translated_label = LABEL_TRANSLATIONS.get(original_label, original_label) # Traduce la etiqueta
                    
                    data.append({
                        "Etiqueta de Objeto": translated_label, # <-- ETIQUETA TRADUCIDA
                        "Confianza (%)": f"{round(score * 100, 2)}%", # Formateado como porcentaje
                        "Coordenadas de la Caja": f"[{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}]" # Formato [x1, y1, x2, y2]
                        # "Área de la Caja (px)": f"{(int(x2)-int(x1)) * (int(y2)-int(y1))} px²" # Podrías añadir el área si es relevante
                    })
                
                # Crear un DataFrame para mostrar en una tabla
                detections_df = pd.DataFrame(data)
                st.dataframe(detections_df)
                
            else:
                st.info("😔 No se detectaron objetos significativos en la imagen.")
                
        except Exception as e:
            st.error(f"❌ Ocurrió un error inesperado al procesar la imagen: {e}")
            st.error("Por favor, verifica el formato de la imagen y los logs de la aplicación para más detalles.")
    else:
        st.info("⬆️ Sube una imagen para que la IA la analice y detecte objetos.")


def settings_section():
    st.title("⚙️ Configuración de la Aplicación")
    st.markdown("""
    Esta sección está reservada para futuras configuraciones y personalizaciones de la aplicación.
    Actualmente, no hay opciones de configuración disponibles.
    """)
    st.info("Próximamente: ¡Nuevas opciones de personalización aquí!")

# --- RUTEO DE SECCIONES ---
if selected == "Predicción Demanda":
    predict_demand_section()
elif selected =
