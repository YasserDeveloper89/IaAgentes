import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
from streamlit_option_menu import option_menu
import plotly.express as px

# --- CONFIGURACIÓN VISUAL ---
PRIMARY_COLOR = "#1f77b4"
ACCENT_COLOR = "#ff7f0e"
BACKGROUND_COLOR = "#f0f2f6"
FONT_FAMILY = "Arial, sans-serif"

st.set_page_config(page_title="AI Agents + Video Analytics", layout="wide", page_icon="🤖")

# --- ESTILOS CSS PARA ESTÉTICA MODERNA ---
st.markdown(f"""
<style>
    /* Fondo y tipografía general */
    .css-1d391kg {{
        background-color: {BACKGROUND_COLOR};
        font-family: {FONT_FAMILY};
    }}
    .stApp {{
        background: linear-gradient(135deg, #e0eafc 0%, #cfdef3 100%);
        color: #333;
        font-family: {FONT_FAMILY};
    }}

    /* Sidebar */
    .sidebar .sidebar-content {{
        background-color: {PRIMARY_COLOR};
        color: white;
        font-family: {FONT_FAMILY};
    }}

    /* Botones */
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

    /* Tablas */
    table {{
        border-collapse: collapse;
        width: 100%;
    }}
    th, td {{
        border: 1px solid #ddd;
        padding: 8px;
        text-align: left;
    }}
    th {{
        background-color: {PRIMARY_COLOR};
        color: white;
    }}
    tr:nth-child(even){{background-color: #f2f2f2;}}

    /* Títulos y subtítulos */
    .stMarkdown h1, .stMarkdown h2 {{
        font-family: {FONT_FAMILY};
        font-weight: 700;
        color: #222;
    }}
</style>
""", unsafe_allow_html=True)

# --- MENU LATERAL CON streamlit-option-menu ---
with st.sidebar:
    selected = option_menu(
        menu_title="Menu Principal",
        options=["Predicción Demanda", "Análisis de Archivos", "Video Analytics", "Configuración"],
        icons=["bar-chart-line", "file-earmark-text", "camera-video", "gear"],
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
    Carga un archivo CSV con las columnas:
    - **fecha** (YYYY-MM-DD)
    - **producto**
    - **cantidad** (vendida)
    """)
    uploaded_file = st.file_uploader("Sube archivo CSV", type=["csv"])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file, parse_dates=['fecha'])
            df = df.sort_values(['producto', 'fecha'])
            st.subheader("Datos Cargados")
            st.dataframe(df)
            
            productos = df['producto'].unique()
            producto_sel = st.selectbox("Selecciona producto", productos)
            
            df_producto = df[df['producto'] == producto_sel].copy()
            
            window = st.slider("Ventana promedio móvil (días)", 2, 10, 3)
            growth_factor = st.slider("Factor de crecimiento esperado", 1.0, 2.0, 1.1, 0.01)
            forecast_days = st.slider("Días a predecir", 1, 14, 7)
            
            df_producto['moving_avg'] = df_producto['cantidad'].rolling(window=window).mean()
            last_avg = df_producto['moving_avg'].iloc[-1]
            
            future_dates = [df_producto['fecha'].iloc[-1] + timedelta(days=i) for i in range(1, forecast_days + 1)]
            forecast_values = [last_avg * (growth_factor ** i) for i in range(1, forecast_days + 1)]
            
            forecast_df = pd.DataFrame({
                'Fecha': future_dates,
                'Cantidad Pronosticada': forecast_values
            })
            
            st.subheader(f"Pronóstico para producto: {producto_sel}")
            
            # Crear DataFrame combinado para gráfico
            df_hist = df_producto[['fecha', 'cantidad']].rename(columns={'fecha': 'Fecha', 'cantidad': 'Cantidad'})
            df_hist['Tipo'] = 'Histórico'
            forecast_df['Tipo'] = 'Pronóstico'
            forecast_df_renamed = forecast_df.rename(columns={'Cantidad Pronosticada': 'Cantidad'})
            
            df_plot = pd.concat([df_hist, forecast_df_renamed], ignore_index=True)
            
            fig = px.line(df_plot, x='Fecha', y='Cantidad', color='Tipo',
                          title="Demanda Histórica y Pronóstico",
                          labels={'Cantidad': 'Cantidad', 'Fecha': 'Fecha', 'Tipo': 'Tipo de dato'})
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### Tabla de Pronóstico")
            st.dataframe(forecast_df.style.format({"Cantidad Pronosticada": "{:.2f}"}))
            
        except Exception as e:
            st.error(f"Error al procesar el archivo: {e}")
    else:
        st.info("Carga un archivo CSV para empezar.")

def file_analysis_section():
    st.title("🔍 Análisis de Archivos")
    st.markdown("Carga un archivo CSV para explorarlo y visualizar estadísticas básicas.")
    uploaded_file = st.file_uploader("Sube archivo CSV", type=["csv"])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.subheader("Vista previa de datos")
            st.dataframe(df.head(10))
            
            st.subheader("Descripción estadística")
            st.write(df.describe(include='all'))
            
            st.subheader("Gráficos rápidos")
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            if numeric_cols:
                col_sel = st.selectbox("Selecciona columna numérica para gráfico", numeric_cols)
                fig = px.histogram(df, x=col_sel, nbins=30, title=f"Histograma de {col_sel}")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No hay columnas numéricas para graficar.")
        except Exception as e:
            st.error(f"Error procesando archivo: {e}")
    else:
        st.info("Sube un archivo CSV para análisis.")

def video_analytics_section():
    st.title("🎥 Video Analytics (Demo)")
    st.markdown("""
    Esta sección está en desarrollo.  
    En producción, aquí se integraría análisis de comportamiento y ocupación a partir de video,  
    pero para evitar problemas con dependencias complejas (cv2) en la nube, esta demo muestra solo placeholder.
    """)
    st.info("Aquí se mostrarían análisis de ocupación, comportamiento de clientes, etc.")

def settings_section():
    st.title("⚙️ Configuración")
    st.markdown("""
    Ajustes generales y configuración de la aplicación.  
    Por ahora, esta sección es estática.
    """)

# --- RUTEO SECCIONES ---
if selected == "Predicción Demanda":
    predict_demand_section()
elif selected == "Análisis de Archivos":
    file_analysis_section()
elif selected == "Video Analytics":
    video_analytics_section()
elif selected == "Configuración":
    settings_section()
else:
    st.write("Selecciona una opción del menú lateral.")
