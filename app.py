import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
from streamlit_option_menu import option_menu
import plotly.express as px

# --- CONFIGURACIÓN VISUAL AVANZADA ---
PRIMARY_COLOR = "#0D3B66"     # Azul oscuro corporativo
ACCENT_COLOR = "#F95738"      # Rojo vibrante
BACKGROUND_COLOR = "#F4F4F9"  # Gris muy claro
CARD_BACKGROUND = "#FFFFFF"   # Blanco para tarjetas
FONT_FAMILY = "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif"
BOX_SHADOW = "0 4px 15px rgba(0,0,0,0.1)"

st.set_page_config(page_title="AI Agents + Video Analytics", layout="wide", page_icon="🤖")

# --- ESTILOS CSS MODERNOS Y AVANZADOS ---
st.markdown(f"""
<style>
    /* Fondo general y fuente */
    .stApp {{
        background: linear-gradient(135deg, #e0eafc 0%, #cfdef3 100%);
        font-family: {FONT_FAMILY};
        color: #222;
    }}

    /* Barra lateral */
    .sidebar .sidebar-content {{
        background-color: {PRIMARY_COLOR};
        color: white;
        font-family: {FONT_FAMILY};
        border-radius: 0 15px 15px 0;
        box-shadow: {BOX_SHADOW};
    }}

    /* Menú lateral opciones */
    .css-1d391kg .stButton>button, .css-1d391kg button {{
        background-color: {ACCENT_COLOR};
        color: white;
        border-radius: 12px;
        padding: 10px 24px;
        font-weight: 600;
        border: none;
        transition: background-color 0.3s ease;
        box-shadow: 0 3px 10px rgba(249, 87, 56, 0.4);
    }}
    .css-1d391kg .stButton>button:hover, .css-1d391kg button:hover {{
        background-color: #d94b2c;
        box-shadow: 0 5px 15px rgba(217, 75, 44, 0.6);
    }}

    /* Inputs, select, sliders */
    div[data-baseweb="select"] > div {{
        border-radius: 12px !important;
        border: 1.5px solid {PRIMARY_COLOR} !important;
        box-shadow: none !important;
        font-weight: 600;
        padding: 0.2rem 0.5rem;
    }}

    .stSlider > div {{
        border-radius: 12px !important;
    }}

    /* Tarjetas */
    .card {{
        background: {CARD_BACKGROUND};
        padding: 20px 30px;
        margin-bottom: 25px;
        border-radius: 18px;
        box-shadow: {BOX_SHADOW};
        transition: transform 0.2s ease;
    }}
    .card:hover {{
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }}

    /* Titulos */
    h1, h2, h3 {{
        font-weight: 700;
        color: {PRIMARY_COLOR};
    }}

    /* Texto informativo */
    .stInfo, .stWarning, .stError {{
        border-radius: 12px;
        padding: 15px;
        font-weight: 600;
    }}

</style>
""", unsafe_allow_html=True)

# --- MENÚ LATERAL MODERNO ---
with st.sidebar:
    selected = option_menu(
        menu_title="Menú Principal",
        options=["Predicción Demanda", "Análisis de Archivos", "Video Analytics", "Configuración"],
        icons=["bar-chart-line", "file-earmark-text", "camera-video", "gear"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "5px", "background-color": PRIMARY_COLOR},
            "icon": {"color": "white", "font-size": "22px"},
            "nav-link": {"font-size": "17px", "text-align": "left", "margin":"0px", "--hover-color": ACCENT_COLOR, "font-weight":"600"},
            "nav-link-selected": {"background-color": ACCENT_COLOR, "font-weight":"700"},
        }
    )

# --- FUNCIONES CON ESTILO Y MEJOR UX ---

def predict_demand_section():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.title("📊 Predicción de Demanda Inteligente")
    st.markdown("""
    Carga un archivo CSV con datos históricos de ventas.  
    Columnas requeridas:  
    - **fecha** (YYYY-MM-DD)  
    - **producto**  
    - **cantidad** (vendida)
    """)
    uploaded_file = st.file_uploader("📂 Selecciona tu archivo CSV aquí", type=["csv"])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file, parse_dates=['fecha'])
            df = df.sort_values(['producto', 'fecha'])
            st.subheader("👀 Datos cargados")
            st.dataframe(df, height=250)
            
            productos = df['producto'].unique()
            producto_sel = st.selectbox("Producto a pronosticar", productos)
            
            df_producto = df[df['producto'] == producto_sel].copy()
            
            window = st.slider("Ventana de promedio móvil (días)", 2, 10, 3)
            growth_factor = st.slider("Factor de crecimiento esperado (1.0 = sin cambio)", 1.0, 2.0, 1.1, 0.01)
            forecast_days = st.slider("Días a predecir", 1, 14, 7)
            
            df_producto['moving_avg'] = df_producto['cantidad'].rolling(window=window).mean()
            last_avg = df_producto['moving_avg'].iloc[-1]
            
            future_dates = [df_producto['fecha'].iloc[-1] + timedelta(days=i) for i in range(1, forecast_days + 1)]
            forecast_values = [last_avg * (growth_factor ** i) for i in range(1, forecast_days + 1)]
            
            forecast_df = pd.DataFrame({'fecha': future_dates, 'Cantidad Pronosticada': forecast_values})
            
            st.subheader(f"📈 Pronóstico para: {producto_sel}")
            combined = pd.concat([
                df_producto.set_index('fecha')['cantidad'],
                forecast_df.set_index('fecha')['Cantidad Pronosticada']
            ])
            fig = px.line(combined, labels={'index':'Fecha', 'value':'Cantidad'},
                          title="Demanda Histórica y Pronóstico")
            fig.update_layout(
                plot_bgcolor='white',
                xaxis=dict(showgrid=True, gridcolor='#f0f0f0'),
                yaxis=dict(showgrid=True, gridcolor='#f0f0f0'),
                font=dict(color=PRIMARY_COLOR)
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### Tabla detallada del pronóstico")
            st.dataframe(forecast_df.style.format({"Cantidad Pronosticada": "{:.2f}"}))
            
        except Exception as e:
            st.error(f"❌ Error procesando el archivo: {e}")
    else:
        st.info("ℹ️ Por favor, sube un archivo CSV para comenzar.")
    st.markdown('</div>', unsafe_allow_html=True)

def file_analysis_section():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.title("🔍 Análisis Avanzado de Archivos CSV")
    st.markdown("""
    Sube un archivo CSV para explorar su contenido y obtener estadísticas y gráficos interactivos.  
    Ideal para entender rápidamente la estructura y valores de tu dataset.
    """)
    
    uploaded_file = st.file_uploader("📂 Selecciona tu archivo CSV aquí", type=["csv"])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            
            st.subheader("👀 Vista previa (Primeras 10 filas)")
            st.dataframe(df.head(10))
            
            st.subheader("📊 Estadísticas básicas")
            st.write("""
            Esta tabla muestra:  
            - Conteo  
            - Media  
            - Desviación estándar  
            - Mínimos y máximos  
            - Valores únicos para columnas no numéricas
            """)
            st.write(df.describe(include='all'))
            
            st.subheader("📈 Visualización de columnas numéricas")
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            if numeric_cols:
                col_sel = st.selectbox("Selecciona columna numérica
