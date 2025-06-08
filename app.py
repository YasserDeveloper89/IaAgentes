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
# Colores Futuristas/Corporativos
BACKGROUND_COLOR = "#0A0A1E" # Azul muy oscuro, casi negro
PRIMARY_COLOR_FUTURISTIC = "#00BCD4" # Cian brillante (como acento principal)
ACCENT_COLOR_FUTURISTIC = "#FF4081" # Rosa/Fucsia (para botones, resaltados)
TEXT_COLOR = "#E0E0E0" # Gris claro para el texto principal
SECONDARY_TEXT_COLOR = "#A0A0B0" # Gris azulado para texto secundario/ayuda
BORDER_COLOR = "#2C2C40" # Gris oscuro para bordes sutiles
FONT_FAMILY = "Segoe UI, Arial, sans-serif" # Fuente moderna

st.set_page_config(
    page_title="Plataforma de IA Corporativa: Soluciones Avanzadas",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="✨"
)

# --- ESTILOS CSS PARA ESTÉTICA ULTRA FUTURISTA Y CORPORATIVA ---
st.markdown(f"""
    <style>
        /* Variables CSS para una fácil personalización */
        :root {{
            --background-color: {BACKGROUND_COLOR};
            --primary-color: {PRIMARY_COLOR_FUTURISTIC};
            --accent-color: {ACCENT_COLOR_FUTURISTIC};
            --text-color: {TEXT_COLOR};
            --secondary-text-color: {SECONDARY_TEXT_COLOR};
            --border-color: {BORDER_COLOR};
            --font-family: {FONT_FAMILY};
        }}

        /* Estilos generales de la aplicación */
        .stApp {{
            background: var(--background-color);
            color: var(--text-color);
            font-family: var(--font-family);
            background-attachment: fixed;
            background-size: cover;
        }}

        /* Sidebar */
        .st-emotion-cache-1d391kg {{ /* Selector para el contenedor principal de Streamlit */
            background-color: var(--background-color); /* Fondo del área principal */
            font-family: var(--font-family);
        }}
        .st-emotion-cache-vk3305 {{ /* Selector específico del sidebar en versiones recientes de Streamlit */
            background-color: #1A1A30; /* Un tono un poco más claro que el fondo para el sidebar */
            color: var(--text-color);
            font-family: var(--font-family);
            box-shadow: 2px 0px 10px rgba(0, 0, 0, 0.5); /* Sombra para dar profundidad */
        }}
        /* Estilo general para todos los botones de Streamlit */
        .stButton>button {{
            background-color: var(--accent-color);
            color: white;
            border-radius: 8px;
            padding: 8px 16px;
            font-weight: bold;
            border: none;
            transition: background-color 0.3s ease, transform 0.2s ease;
            cursor: pointer;
        }}
        .stButton>button:hover {{
            background-color: #E6007A; /* Tono más oscuro del acento al hover */
            transform: translateY(-2px);
        }}

        /* Títulos */
        h1, h2, h3, h4, h5, h6 {{
            color: var(--primary-color);
            font-weight: 700;
            margin-bottom: 0.8em;
            text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.3);
        }}
        h1 {{
            text-align: center;
            font-size: 2.5em;
            letter-spacing: 1.5px;
            padding-bottom: 0.5em;
            border-bottom: 2px solid var(--border-color);
        }}
        h3 {{
            color: var(--primary-color);
            font-weight: 600;
            margin-top: 1.5em;
            margin-bottom: 0.7em;
            border-left: 5px solid var(--accent-color);
            padding-left: 10px;
        }}

        /* Texto y Markdown */
        p, li {{
            color: var(--text-color);
            font-size: 1.05em;
            line-height: 1.6;
        }}
        .stMarkdown {{
            color: var(--text-color);
        }}
        .stMarkdown a {{
            color: var(--accent-color);
            text-decoration: none;
        }}
        .stMarkdown a:hover {{
            text-decoration: underline;
        }}

        /* Widgets de Streamlit */
        .stTextInput>div>div>input, .stSelectbox>div>div, .stSlider>div>div>div, .stRadio>label {{
            background-color: #1A1A30; /* Fondo más oscuro para inputs */
            color: var(--text-color);
            border: 1px solid var(--border-color);
            border-radius: 5px;
            padding: 8px 12px;
            font-size: 1em;
            box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.3);
        }}
        .stTextInput>div>div>input:focus, .stSelectbox>div:focus-within>div, .stSlider>div>div>div:focus-within, .stRadio>label:hover {{
            border-color: var(--primary-color);
            box-shadow: 0 0 0 0.2rem rgba(0, 188, 212, 0.25); /* Resplandor al enfocar */
        }}
        .stSelectbox>div>div:hover, .stRadio>label:hover {{
            background-color: #2A2A40;
        }}
        .stSlider .stSliderVertical > div > div {{ /* Color del relleno del slider */
            background-color: var(--primary-color);
        }}
        .stSlider .stSliderVertical > div > div > div {{ /* Pulgar del slider */
            background-color: var(--accent-color);
            border: 2px solid var(--text-color);
        }}

        /* Mensajes de alerta */
        .stAlert {{
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 20px;
            border: 1px solid var(--border-color);
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.4);
            background-color: #1A1A30;
        }}
        .stAlert.info {{
            color: var(--primary-color);
            border-left: 5px solid var(--primary-color);
        }}
        .stAlert.warning {{
            color: #FFC107; /* Amarillo para advertencias */
            border-left: 5px solid #FFC107;
        }}
        .stAlert.error {{
            color: #F44336; /* Rojo para errores */
            border-left: 5px solid #F44336;
        }}
        .stAlert.success {{
            color: #4CAF50; /* Verde para éxito */
            border-left: 5px solid #4CAF50;
        }}

        /* Tablas (DataFrames) */
        .dataframe {{
            font-size: 0.95em;
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 20px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.4);
            border-radius: 8px;
            overflow: hidden; /* Para que los bordes redondeados se vean bien */
        }}
        .dataframe th, .dataframe td {{
            padding: 12px 15px;
            border: 1px solid var(--border-color);
            text-align: left;
            color: var(--text-color);
        }}
        .dataframe th {{
            background-color: var(--primary-color);
            color: #FFFFFF;
            font-weight: bold;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .dataframe tbody tr:nth-child(even) {{
            background-color: #1A1A30; /* Filas pares con un fondo ligeramente diferente */
        }}
        .dataframe tbody tr:nth-child(odd) {{
            background-color: #101025; /* Filas impares */
        }}
        .dataframe tbody tr:hover {{
            background-color: #252540; /* Efecto hover en filas */
            transition: background-color 0.3s ease;
        }}

        /* Botones del menú lateral */
        [data-testid="stSidebarNav"] li a {{
            color: var(--text-color);
            font-size: 1.1em;
            padding: 10px 15px;
            margin: 5px 0;
            border-radius: 8px;
            transition: background-color 0.3s ease, color 0.3s ease;
        }}
        [data-testid="stSidebarNav"] li a:hover {{
            background-color: var(--border-color);
            color: var(--primary-color);
        }}
        [data-testid="stSidebarNav"] li a.active {{ /* Elemento seleccionado del menú */
            background-color: var(--primary-color);
            color: #FFFFFF;
            font-weight: bold;
        }}
        [data-testid="stSidebarNav"] .css-1lcbm67 {{ /* Título del menú */
            color: var(--primary-color) !important;
            font-size: 1.6em !important;
            font-weight: 700 !important;
            text-align: center !important;
            margin-bottom: 1.5em !important;
            padding-bottom: 0.5em !important;
            border-bottom: 2px solid var(--border-color) !important;
        }}

    </style>
""", unsafe_allow_html=True)


# --- Diccionario para traducir las etiquetas de YOLOv8 a español ---
# Este diccionario se usará para la tabla de detecciones.
# Asegúrate de añadir aquí las clases específicas que quieras para YOLO-World y sus traducciones.
LABEL_TRANSLATIONS = {
    'person': 'Persona', 'bicycle': 'Bicicleta', 'car': 'Coche', 'motorcycle': 'Motocicleta',
    'airplane': 'Avión', 'bus': 'Autobús', 'train': 'Tren', 'truck': 'Camión',
    'boat': 'Barco', 'traffic light': 'Semáforo', 'fire hydrant': 'Boca de Incendios',
    'stop sign': 'Señal de Stop', 'parking meter': 'Parquímetro', 'bench': 'Banco',
    'bird': 'Pájaro', 'cat': 'Gato', 'dog': 'Perro', 'horse': 'Caballo', 'sheep': 'Oveja',
    'cow': 'Vaca', 'elephant': 'Elefante', 'bear': 'Oso', 'zebra': 'Cebra', 'giraffe': 'Jirafa',
    'backpack': 'Mochila', 'umbrella': 'Paraguas', 'handbag': 'Bolso', 'tie': 'Corbata',
    'suitcase': 'Maleta', 'frisbee': 'Frisbee', 'skis': 'Esquís', 'snowboard': 'Tabla de Snow',
    'sports ball': 'Pelota de Deporte', 'kite': 'Cometa', 'baseball bat': 'Bate de Béisbol',
    'baseball glove': 'Guante de Béisbol', 'skateboard': 'Monopatín', 'surfboard': 'Tabla de Surf',
    'tennis racket': 'Raqueta de Tenis', 'bottle': 'Botella', 'wine glass': 'Copa de Vino',
    'cup': 'Taza', 'fork': 'Tenedor', 'knife': 'Cuchillo', 'spoon': 'Cuchara', 'bowl': 'Cuenco',
    'banana': 'Plátano', 'apple': 'Manzana', 'sandwich': 'Sándwich', 'orange': 'Naranja',
    'broccoli': 'Brócoli', 'hot dog': 'Perrito Caliente', 'pizza': 'Pizza', 'donut': 'Dona',
    'cake': 'Tarta', 'chair': 'Silla', 'couch': 'Sofá', 'potted plant': 'Planta en Maceta',
    'bed': 'Cama', 'dining table': 'Mesa de Comedor', 'toilet': 'Inodoro', 'tv': 'Televisión',
    'laptop': 'Portátil', 'mouse': 'Ratón', 'remote': 'Mando a Distancia', 'keyboard': 'Teclado',
    'cell phone': 'Teléfono Móvil', 'microwave': 'Microondas', 'oven': 'Horno', 'toaster': 'Tostadora',
    'sink': 'Fregadero', 'refrigerator': 'Refrigerador', 'book': 'Libro', 'clock': 'Reloj',
    'vase': 'Jarrón', 'scissors': 'Tijeras', 'teddy bear': 'Oso de Peluche',
    'hair drier': 'Secador de Pelo', 'toothbrush': 'Cepillo de Dientes',
    # --- Clases específicas para Restaurantes y Clínicas (ejemplos para YOLO-World) ---
    'fresa': 'Fresa', 'uva': 'Uva', 'kiwi': 'Kiwi', 'plato': 'Plato', 'cubierto': 'Cubierto',
    'bandeja': 'Bandeja', 'vaso': 'Vaso', 'comida': 'Comida', 'ingrediente': 'Ingrediente',
    'salsa': 'Salsa', 'chef hat': 'Gorro de Chef', 'delantal': 'Delantal', 'menu': 'Menú',
    'jeringa': 'Jeringa', 'mascarilla': 'Mascarilla', 'guantes medicos': 'Guantes Médicos',
    'venda': 'Venda', 'estetoscopio': 'Estetoscopio', 'termometro': 'Termómetro',
    'medicamento': 'Medicamento', 'pastilla': 'Pastilla', 'instrumento quirurgico': 'Instrumento Quirúrgico',
    'botiquin': 'Botiquín', 'equipo medico': 'Equipo Médico', 'camilla': 'Camilla'
}

# --- SELECCIÓN DEL TIPO DE NEGOCIO (GLOBAL) ---
# Usaremos st.session_state para almacenar la elección del usuario
if 'business_type' not in st.session_state:
    st.session_state.business_type = None

# Define business_options AQUÍ, antes de cualquier uso en la lógica principal
business_options = {
    "Restaurante": "🍽️ Soluciones para la gestión culinaria y de clientes.",
    "Clínica": "🏥 Optimización de procesos sanitarios y atención al paciente."
}

# Mostrar la selección solo al inicio o si no se ha elegido
if st.session_state.business_type is None:
    st.title("Bienvenido a la Plataforma de IA Corporativa")
    st.markdown("""
        Esta plataforma está diseñada para potenciar tu negocio con herramientas avanzadas de Inteligencia Artificial.
        Para ofrecerte una experiencia personalizada y relevante, por favor, selecciona el tipo de negocio que representas.
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Soy un Restaurante", key="btn_restaurant", use_container_width=True):
            st.session_state.business_type = "Restaurante"
            st.rerun() # Reruns the app to apply changes
    with col2:
        if st.button("Soy una Clínica", key="btn_clinic", use_container_width=True):
            st.session_state.business_type = "Clínica"
            st.rerun() # Reruns the app to apply changes
    
    # IMPORTANTE: No usamos st.stop() aquí. Si se llega a este punto
    # y no se ha seleccionado nada, la ejecución simplemente termina
    # y Streamlit vuelve a renderizar cuando se haga una selección.
    # El st.rerun() se encargará de reiniciar la aplicación y cargar el contenido principal.
    
else: # Esta sección solo se ejecuta si st.session_state.business_type NO es None
    # --- Interfaz principal de la aplicación (Sidebar y Módulos) ---

    st.sidebar.title(f"Tipo de Negocio: {st.session_state.business_type} {business_options[st.session_state.business_type].split(' ')[0]}")
    if st.sidebar.button("Cambiar tipo de negocio", help="Vuelve a la pantalla de selección inicial", key="sidebar_change_type"):
        st.session_state.business_type = None
        st.rerun()

    # --- MENÚ LATERAL ---
    with st.sidebar:
        st.markdown("---") # Separador visual
        selected = option_menu(
            menu_title="Módulos de IA", # Título del menú
            options=["Predicción Demanda", "Análisis Archivos", "Análisis de Imágenes", "Configuración"],
            icons=["bar-chart-line", "file-earmark-text", "image", "gear"], # Iconos modernos
            menu_icon="cast", # Icono del menú principal
            default_index=0,
            styles={
                "container": {"padding": "5px", "background-color": "#1A1A30"}, # Fondo del contenedor del menú
                "icon": {"color": PRIMARY_COLOR_FUTURISTIC, "font-size": "20px"},
                "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": BORDER_COLOR, "color": TEXT_COLOR},
                "nav-link-selected": {"background-color": PRIMARY_COLOR_FUTURISTIC, "color": "#FFFFFF"},
            }
        )

    # --- Funciones de las secciones (definiciones) ---
    # Nota: Las funciones deben estar definidas a nivel global para ser invocables.
    # Las definiciones completas de estas funciones están en la Parte 2.

    def predict_demand_section():
        pass # Contenido en la Parte 2

    def file_analysis_section():
        pass # Contenido en la Parte 2

    def image_analysis_section():
        pass # Contenido en la Parte 2

    def settings_section():
        pass # Contenido en la Parte 2





def predict_demand_section():
    business_type = st.session_state.business_type

    if business_type == "Restaurante":
        st.title("📊 Predicción de Demanda - Optimización Culinaria")
        st.markdown("""
        Este módulo te permite pronosticar la demanda de platos, ingredientes o el flujo de clientes en tu restaurante.
        Una predicción precisa ayuda a optimizar el inventario, reducir el desperdicio y mejorar la planificación del personal.
        
        Por favor, sube un archivo CSV con las siguientes columnas:
        - 📅 **fecha** (formato:YYYY-MM-DD)  
        - 🍽️ **elemento** (ej: "Pizza Margarita", "Ingrediente A", "Número de Clientes")  
        - 🔢 **cantidad** (unidades vendidas o recuento de clientes/reservas)

        **Instrucciones de Uso:**
        1. Sube tu archivo CSV con los datos históricos de tu restaurante.
        2. Selecciona el 'elemento' (plato, ingrediente, etc.) para el cual deseas generar el pronóstico.
        3. Ajusta los parámetros de predicción para refinar los resultados según tus necesidades.
        """)
    elif business_type == "Clínica":
        st.title("📊 Predicción de Demanda - Gestión Sanitaria Eficiente")
        st.markdown("""
        Este módulo está diseñado para pronosticar la demanda de servicios médicos, insumos específicos o la afluencia de pacientes en tu clínica.
        Mejora la planificación de recursos, la gestión de inventario de suministros críticos y la programación de personal.
        
        Por favor, sube un archivo CSV con las siguientes columnas:
        - 📅 **fecha** (formato:YYYY-MM-DD)  
        - 🏥 **elemento** (ej: "Consulta General", "Vacuna X", "Número de Pacientes")  
        - 🔢 **cantidad** (número de servicios prestados o insumos utilizados)

        **Instrucciones de Uso:**
        1. Sube tu archivo CSV con los datos históricos de tu clínica.
        2. Selecciona el 'elemento' (servicio, insumo, etc.) para el cual deseas generar el pronóstico.
        3. Ajusta los parámetros de predicción para refinar los resultados según tus necesidades operativas.
        """)
    
    uploaded_file = st.file_uploader("Sube tu archivo CSV de datos históricos aquí", type=["csv"])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file, parse_dates=['fecha'])
            # Asegúrate de que la columna 'elemento' exista antes de usarla
            if 'elemento' not in df.columns:
                # Intenta renombrar 'producto' a 'elemento' si existe, para compatibilidad
                if 'producto' in df.columns:
                    df.rename(columns={'producto': 'elemento'}, inplace=True)
                else:
                    st.error("El archivo CSV debe contener las columnas 'fecha', 'elemento' y 'cantidad'.")
                    return # Detener la ejecución si faltan columnas
            
            df = df.sort_values(['elemento', 'fecha'])

            st.subheader("✅ Datos históricos cargados correctamente")
            st.dataframe(df)

            elementos = df['elemento'].unique()
            elemento_sel = st.selectbox("👉 Selecciona el elemento para el cual deseas predecir la demanda", elementos)

            df_elemento = df[df['elemento'] == elemento_sel].copy()

            st.subheader("⚙️ Parámetros de Predicción")
            window = st.slider("Ventana para promedio móvil (número de días a considerar para el promedio)", 2, 10, 3, help="Define cuántos días anteriores se usan para calcular el promedio móvil. Un valor más alto suaviza las fluctuaciones.")
            growth_factor = st.slider("Factor de crecimiento esperado (ej: 1.05 para un 5% de crecimiento diario)", 1.0, 2.0, 1.02, 0.01, help="Establece el factor por el cual se espera que la demanda crezca cada día en el futuro. 1.0 significa sin crecimiento.")
            forecast_days = st.slider("Días a predecir (número de días futuros para el pronóstico)", 1, 30, 7, help="Determina cuántos días hacia adelante deseas pronosticar la demanda.")

            df_elemento['promedio_movil'] = df_elemento['cantidad'].rolling(window=window).mean().round(2)
            
            if not df_elemento['promedio_movil'].dropna().empty:
                last_avg = df_elemento['promedio_movil'].iloc[-1]
            else:
                st.warning("No hay suficientes datos para calcular el promedio móvil. Ajusta la ventana o el dataset.")
                last_avg = df_elemento['cantidad'].mean() if not df_elemento['cantidad'].empty else 0
                if last_avg == 0:
                    st.error("No hay datos de cantidad válidos para realizar una predicción.")
                    return # Detener si no hay datos de cantidad

                st.info(f"Usando el promedio simple de cantidad ({last_avg:.2f}) para la predicción debido a datos insuficientes para el promedio móvil.")


            future_dates = [df_elemento['fecha'].iloc[-1] + timedelta(days=i) for i in range(1, forecast_days + 1)]
            forecast_values = []
            current_forecast_value = last_avg
            for i in range(forecast_days):
                current_forecast_value *= growth_factor
                forecast_values.append(round(current_forecast_value))

            forecast_df = pd.DataFrame({'fecha': future_dates, 'cantidad_prevista': forecast_values})

            st.subheader(f"📈 Pronóstico de Demanda para: **{elemento_sel}**")

            combined = pd.concat([
                df_elemento.set_index('fecha')['cantidad'].rename('Cantidad Histórica'),
                forecast_df.set_index('fecha')['cantidad_prevista'].rename('Cantidad Prevista')
            ], axis=1).reset_index()
            
            fig = px.line(
                combined, 
                x='fecha', 
                y=['Cantidad Histórica', 'Cantidad Prevista'],
                title=f'Análisis y Pronóstico de Demanda para {elemento_sel}',
                labels={'value': 'Cantidad', 'fecha': 'Fecha', 'variable': 'Tipo de Cantidad'},
                color_discrete_map={
                    'Cantidad Histórica': PRIMARY_COLOR_FUTURISTIC, 
                    'Cantidad Prevista': ACCENT_COLOR_FUTURISTIC
                },
                template="plotly_white"
            )
            
            fig.update_traces(mode='lines+markers', hovertemplate="Fecha: %{x}<br>Cantidad: %{y}<extra></extra>")
            fig.update_layout(
                hovermode="x unified",
                xaxis_title="Fecha",
                yaxis_title="Cantidad de Unidades / Registros",
                legend_title="Leyenda",
                font=dict(family=FONT_FAMILY, color=TEXT_COLOR),
                margin=dict(l=0, r=0, t=50, b=0),
                paper_bgcolor=BACKGROUND_COLOR,
                plot_bgcolor='#1A1A30'
            )
            if not df_elemento.empty:
                last_historical_date = df_elemento['fecha'].iloc[-1]
                fig.add_vline(x=last_historical_date, line_width=2, line_dash="dash", line_color=SECONDARY_TEXT_COLOR, 
                              annotation_text="Inicio Pronóstico", annotation_position="top right")

            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### 📋 Tabla Detallada del Pronóstico")
            st.dataframe(forecast_df.rename(columns={'fecha': 'Fecha', 'cantidad_prevista': 'Cantidad Prevista'}).set_index('Fecha'))
            
        except pd.errors.EmptyDataError:
            st.error("El archivo CSV está vacío. Por favor, sube un archivo con datos.")
        except KeyError as ke:
            st.error(f"Faltan columnas requeridas en tu CSV. Asegúrate de tener 'fecha', 'elemento' y 'cantidad'. Error: {ke}")
        except Exception as e:
            st.error(f"Ocurrió un error inesperado al procesar el archivo: {e}")
            st.info("Asegúrate de que el formato de la fecha sea `%Y-%m-%d` y que todas las columnas necesarias estén presentes.")
    else:
        st.info("⬆️ Carga tu archivo CSV para comenzar el análisis de predicción de demanda.")


def file_analysis_section():
    business_type = st.session_state.business_type

    if business_type == "Restaurante":
        st.title("🔍 Análisis Exploratorio de Archivos CSV - Visión Profunda del Negocio")
        st.markdown("""
        Analiza tus datos de ventas, inventario, personal o encuestas de clientes para identificar patrones y oportunidades en tu restaurante.
        Sube un archivo CSV para visualizar sus datos, estadísticas descriptivas y la distribución de sus columnas.
        """)
    elif business_type == "Clínica":
        st.title("🔍 Análisis Exploratorio de Archivos CSV - Inteligencia Sanitaria")
        st.markdown("""
        Explora tus registros de pacientes (anonimizados), datos de citas, consumo de insumos o resultados de encuestas de satisfacción para mejorar la gestión clínica.
        Sube un archivo CSV para visualizar sus datos, estadísticas descriptivas y la distribución de sus columnas.
        """)

    uploaded_file = st.file_uploader("Sube tu archivo CSV aquí para análisis exploratorio", type=["csv"])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            
            st.subheader("✅ Vista Previa de los Datos (Primeras 10 filas)")
            st.dataframe(df.head(10))
            
            st.subheader("📊 Estadísticas Descriptivas Generales")
            st.write(df.describe(include='all').T.round(2))
            
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            if numeric_cols:
                st.subheader("📈 Visualización de Distribuciones Numéricas")
                col_sel = st.selectbox("👉 Selecciona una columna numérica para visualizar su distribución (Histograma)", numeric_cols)
                
                fig = px.histogram(df, x=col_sel, nbins=30, 
                                   title=f"Distribución de Frecuencia de '{col_sel}'",
                                   labels={col_sel: col_sel, 'count': 'Frecuencia'},
                                   template="plotly_white",
                                   color_discrete_sequence=[PRIMARY_COLOR_FUTURISTIC])
                
                fig.update_layout(
                    xaxis_title=col_sel,
                    yaxis_title="Frecuencia de Aparición",
                    font=dict(family=FONT_FAMILY, color=TEXT_COLOR),
                    margin=dict(l=0, r=0, t=50, b=0),
                    paper_bgcolor=BACKGROUND_COLOR,
                    plot_bgcolor='#1A1A30'
                )
                st.plotly_chart(fig, use_container_width=True)

                st.subheader(f"📦 Detección de Valores Atípicos para '{col_sel}'")
                fig_box = px.box(df, y=col_sel, 
                                 title=f"Diagrama de Caja de '{col_sel}' (Identificación de Valores Atípicos)",
                                 labels={col_sel: col_sel},
                                 template="plotly_white",
                                 color_discrete_sequence=[ACCENT_COLOR_FUTURISTIC])
                fig_box.update_layout(
                    yaxis_title=col_sel,
                    font=dict(family=FONT_FAMILY, color=TEXT_COLOR),
                    margin=dict(l=0, r=0, t=50, b=0),
                    paper_bgcolor=BACKGROUND_COLOR,
                    plot_bgcolor='#1A1A30'
                )
                st.plotly_chart(fig_box, use_container_width=True)

            else:
                st.info("ℹ️ No se encontraron columnas numéricas en el archivo para generar gráficos de distribución.")
            
        except pd.errors.EmptyDataError:
            st.error("El archivo CSV está vacío. Por favor, sube un archivo con datos.")
        except Exception as e:
            st.error(f"Ocurrió un error inesperado al procesar el archivo: {e}")
            st.info("Asegúrate de que el archivo es un CSV válido y no está corrupto.")
    else:
        st.info("⬆️ Sube un archivo CSV para comenzar el análisis exploratorio de tus datos.")


def image_analysis_section():
    business_type = st.session_state.business_type

    if business_type == "Restaurante":
        st.title("📸 Análisis Inteligente de Imágenes - Visión para Restaurantes")
        st.markdown("""
        Utiliza esta herramienta para la detección de objetos en tu restaurante.
        Puedes identificar inventario (botellas, ingredientes), control de calidad de alimentos, o incluso elementos de seguridad e higiene.
        Sube una imagen y la IA te proporcionará una detección detallada.
        """)
    elif business_type == "Clínica":
        st.title("📸 Análisis Inteligente de Imágenes - Visión para Clínicas")
        st.markdown("""
        Este módulo aplica IA para detectar objetos en imágenes de tu entorno clínico.
        Útil para gestión de instrumental, verificación de stock de insumos médicos o control de limpieza en áreas específicas.
        Sube una imagen y la IA te proporcionará una detección detallada.
        """)

    st.info("Selecciona 'YOLO-World (vocabulario abierto)' para detectar objetos específicos de tu negocio (ej: fresa, jeringa).")
    
    # --- Opción para seleccionar el tipo de modelo ---
    model_choice = st.radio(
        "👉 Selecciona el tipo de modelo para la detección:",
        ('YOLOv8 General (objetos comunes)', 'YOLO-World (vocabulario abierto)'),
        key="model_selection", # Añadir key para evitar DuplicateWidgetID
        help="YOLOv8 General es rápido y eficaz para objetos cotidianos (personas, coches). YOLO-World permite especificar objetos por texto (ej: instrumental médico, frutas específicas), pero puede ser más lento."
    )

    custom_classes = []
    if model_choice == 'YOLO-World (vocabulario abierto)':
        default_classes_text = ""
        if business_type == "Restaurante":
            default_classes_text = "fresa, uva, plátano, tomate, plato, cubierto, botella de vino, vaso, menú"
        elif business_type == "Clínica":
            default_classes_text = "jeringa, mascarilla, guantes medicos, estetoscopio, medicamento, termometro, camilla, botiquin"

        st.info(f"Para YOLO-World, especifica los objetos que deseas detectar, separados por comas. Ejemplos para {business_type}: **{default_classes_text}**")
        custom_classes_input = st.text_input(
            "Objetos personalizados a detectar (ej: fresa, jeringa, botella de aceite):",
            value=default_classes_text,
            key="custom_classes_input" # Añadir key
        )
        if custom_classes_input:
            custom_classes = [c.strip().lower() for c in custom_classes_input.split(',') if c.strip()]
            if not custom_classes:
                st.warning("No se especificaron objetos. YOLO-World puede no detectar lo que esperas.")
        else:
            st.warning("Por favor, introduce al menos un objeto para que YOLO-World lo detecte.")


    uploaded_file = st.file_uploader("Sube tu imagen aquí (formatos soportados: JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"], key="image_uploader")
    if uploaded_file:
        try: # Este 'try' envuelve TODO el procesamiento de la imagen para capturar errores
            st.info("✅ Imagen subida. Preparando para el análisis...")
            img_bytes = uploaded_file.read()
            img = Image.open(io.BytesIO(img_bytes))
            st.image(img, caption="Imagen Original Cargada", use_container_width=True)
            
            model = None
            if model_choice == 'YOLOv8 General (objetos comunes)':
                st.info("⚙️ Cargando el modelo de detección de objetos YOLOv8n (general)...")
                model = YOLO('yolov8n.pt') # O 'yolov8s.pt' / 'yolov8m.pt' para más precisión general
                st.info("✅ Modelo YOLOv8 General cargado.")
            elif model_choice == 'YOLO-World (vocabulario abierto)':
                st.info("⚙️ Cargando el modelo de detección de objetos YOLO-World (vocabulario abierto)...")
                # Se usará yolov8s-world.pt que es un buen balance entre tamaño y capacidad
                model = YOLO('yolov8s-world.pt') 
                if custom_classes:
                    model.set_classes(custom_classes)
                    st.info(f"YOLO-World configurado para detectar: {', '.join(custom_classes)}.")
                else:
                    st.warning("YOLO-World se cargó, pero no se especificaron clases personalizadas. Puede que no detecte lo que esperas.")
                st.info("✅ Modelo YOLO-World cargado.")

            if model:
                st.info("✨ Iniciando detección de objetos...")
                results = model(img)
                st.info("✨ Detección de objetos completada.")
                
                st.subheader("🖼️ Imagen con Objetos Detectados")
                res_img_array = results[0].plot() # Esto es un array de NumPy
                st.image(res_img_array, caption="Objetos Detectados por la IA", use_container_width=True)
                
                st.markdown("### 📋 Detalles de los Objetos Detectados")
                detections = results[0].boxes.data.cpu().numpy() if results[0].boxes is not None else []
                
                if len(detections) > 0:
                    labels = results[0].names
                    data = []
                    for box in detections:
                        x1, y1, x2, y2, score, class_id = box
                        original_label = labels[int(class_id)]
                        
                        translated_label = LABEL_TRANSLATIONS.get(original_label.lower(), original_label)
                        
                        data.append({
                            "Etiqueta de Objeto": translated_label,
                            "Confianza (%)": f"{round(score * 100, 2)}%",
                            "Coordenadas de la Caja": f"[{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}]"
                        })
                    
                    detections_df = pd.DataFrame(data)
                    st.dataframe(detections_df)
                    
                else:
                    st.info("😔 No se detectaron objetos significativos en la imagen con el modelo seleccionado. Intenta con otra imagen o ajusta los parámetros de detección.")
                    
        except Exception as e: 
            st.error(f"❌ Ocurrió un error inesperado al procesar la imagen: {e}")
            st.error("Por favor, verifica el formato de la imagen, los objetos especificados para YOLO-World y los logs de la aplicación para más detalles.")
    else: 
        st.info("⬆️ Sube una imagen para que la IA la analice y detecte objetos.")


def settings_section(): 
    st.title("⚙️ Configuración del Sistema")
    st.markdown("""
    Esta sección permite la configuración y personalización avanzada de la plataforma.
    Actualmente, no hay opciones de configuración disponibles para el usuario final.
    """)
    st.info("Próximamente: ¡Nuevas opciones de personalización y gestión de módulos aquí!")


    # --- RUTEO DE SECCIONES ---
    # Aquí se invocarán las funciones definidas en la Parte 2
    if selected == "Predicción Demanda":
        predict_demand_section()
    elif selected == "Análisis Archivos":
        file_analysis_section()
    elif selected == "Análisis de Imágenes":
        image_analysis_section()
    elif selected == "Configuración":
        settings_section()
    else:
        st.write("Selecciona un módulo del menú para comenzar.")

