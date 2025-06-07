import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import datetime

st.set_page_config(page_title="AI Agents + Video Analytics", layout="wide", page_icon="🤖")

# --- ESTILOS CSS ---
st.markdown("""
<style>
body, .css-18e3th9 {
    background-color: #0d1117;
    color: #c9d1d9;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
h1, h2, h3 {
    color: #58a6ff;
}
.stButton>button {
    background-color: #238636;
    color: white;
    border-radius: 8px;
    height: 38px;
    width: 100%;
    font-weight: 600;
}
.stButton>button:hover {
    background-color: #2ea043;
    color: white;
}
.css-1d391kg {
    background-color: #161b22;
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 8px 24px rgb(20 20 20 / 0.9);
}
</style>
""", unsafe_allow_html=True)

# --- MENU LATERAL MODERNO ---
with st.sidebar:
    selected = option_menu(
        menu_title="Menú Principal",
        options=["Predicción Demanda", "Análisis Archivos", "Resumen Avanzado", "Sobre Nosotros"],
        icons=["graph-up", "file-earmark-text", "bar-chart", "info-circle"],
        menu_icon="robot",
        default_index=0,
        styles={
            "container": {"background-color": "#0d1117", "padding": "10px"},
            "icon": {"color": "#58a6ff", "font-size": "20px"},
            "nav-link": {"font-size": "16px", "color": "#c9d1d9", "hover-color": "#58a6ff"},
            "nav-link-selected": {"background-color": "#238636", "color": "white"},
        }
    )

st.title("🤖 AI Agents + Video Analytics")

# ===========================
# FUNCIONES PRINCIPALES
# ===========================

def predict_demand_interface():
    st.header("Predicción de Demanda")
    st.markdown("Sube tu archivo CSV con ventas históricas para predecir la demanda futura.")
    uploaded_file = st.file_uploader("Carga archivo CSV (fecha, producto, cantidad)", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("Datos cargados:")
            st.dataframe(df.head())

            # Validar columnas mínimas
            required_cols = {'fecha', 'producto', 'cantidad'}
            if not required_cols.issubset(df.columns.str.lower()):
                st.error(f"El CSV debe contener las columnas: {required_cols}")
                return

            df.columns = df.columns.str.lower()
            df['fecha'] = pd.to_datetime(df['fecha'])

            # Simple predicción: demanda promedio por producto * factor crecimiento
            growth_factor = 1.1
            forecast_days = st.slider("Días a predecir", 1, 30, 7)

            last_date = df['fecha'].max()
            future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, forecast_days+1)]

            grouped = df.groupby('producto')['cantidad'].mean().reset_index()
            grouped['predicted'] = grouped['cantidad'] * growth_factor

            forecast = pd.DataFrame()
            for date in future_dates:
                temp = grouped[['producto', 'predicted']].copy()
                temp['fecha'] = date
                forecast = pd.concat([forecast, temp])

            st.markdown("### Predicción de demanda para los próximos días:")
            st.dataframe(forecast.rename(columns={'predicted':'Cantidad estimada'}).style.format({"Cantidad estimada":"{:.2f}"}))

        except Exception as e:
            st.error(f"Error procesando el archivo: {e}")
    else:
        st.info("Carga un archivo CSV para iniciar.")

def analyze_files_interface():
    st.header("Análisis Inteligente de Archivos")
    st.markdown("Carga archivos CSV para análisis estadístico y visualización avanzada.")
    uploaded_file = st.file_uploader("Carga archivo CSV para análisis", type=["csv"])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("Datos cargados:")
            st.dataframe(df.head())

            st.markdown("### Estadísticas Descriptivas")
            st.write(df.describe())

            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            if numeric_cols:
                st.markdown("### Gráficos")
                for col in numeric_cols:
                    st.line_chart(df[col])
            else:
                st.info("No hay columnas numéricas para graficar.")

        except Exception as e:
            st.error(f"Error procesando el archivo: {e}")
    else:
        st.info("Carga un archivo para análisis.")

def advanced_summary_interface():
    st.header("Resumen Avanzado")
    st.markdown("Vista rápida con análisis estadístico, gráficos y resumen de tendencias.")

    # Generar datos demo
    dates = pd.date_range(end=datetime.date.today(), periods=90)
    data = pd.DataFrame({
        "fecha": np.tile(dates, 3),
        "producto": np.repeat(["Café", "Leche", "Harina"], len(dates)),
        "cantidad": np.random.poisson(lam=20, size=len(dates)*3)
    })

    grouped = data.groupby(['fecha', 'producto'])['cantidad'].sum().reset_index()
    st.write("Datos simulados:")
    st.dataframe(grouped.head())

    st.markdown("### Total por Producto")
    prod_totals = grouped.groupby('producto')['cantidad'].sum()
    st.bar_chart(prod_totals)

    st.markdown("### Tendencias de consumo")
    selected_product = st.selectbox("Selecciona producto", prod_totals.index.tolist())
    prod_data = grouped[grouped['producto'] == selected_product].set_index('fecha')
    st.line_chart(prod_data['cantidad'])

def about_us_interface():
    st.header("Sobre Nosotros")
    st.markdown("""
    **AI Agents + Video Analytics**  
    Innovamos con inteligencia artificial para optimizar restaurantes y clínicas.  
    Nuestro sistema predice demanda, optimiza inventarios y analiza comportamiento de clientes/pacientes.  
  
    Contáctanos para soluciones personalizadas.  
    - Email: contacto@iaagentes.com  
    - Teléfono: +51 987 654 321  
    - Web: www.iaagentes.com
    """)

# --- RENDER SECCIONES ---
if selected == "Predicción Demanda":
    predict_demand_interface()
elif selected == "Análisis Archivos":
    analyze_files_interface()
elif selected == "Resumen Avanzado":
    advanced_summary_interface()
elif selected == "Sobre Nosotros":
    about_us_interface()
