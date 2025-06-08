
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
BACKGROUND_COLOR = "#0A0A1E"
PRIMARY_COLOR_FUTURISTIC = "#00BCD4"
ACCENT_COLOR_FUTURISTIC = "#FF4081"
TEXT_COLOR = "#E0E0E0"
SECONDARY_TEXT_COLOR = "#A0A0B0"
BORDER_COLOR = "#2C2C40"
FONT_FAMILY = "Segoe UI, Arial, sans-serif"

st.set_page_config(
    page_title="Plataforma de IA Corporativa: Soluciones Avanzadas",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="✨"
)

# (Se omite el CSS para brevedad, pero debe incluirse en tu versión completa)

# --- Diccionario para traducir las etiquetas de YOLOv8 a español ---
LABEL_TRANSLATIONS = {
    'person': 'Persona', 'bicycle': 'Bicicleta', 'car': 'Coche', 'motorcycle': 'Motocicleta',
    'dog': 'Perro', 'cat': 'Gato', 'bottle': 'Botella', 'cup': 'Taza', 'chair': 'Silla',
    'jeringa': 'Jeringa', 'mascarilla': 'Mascarilla', 'guantes medicos': 'Guantes Médicos',
    'fresa': 'Fresa', 'uva': 'Uva', 'plato': 'Plato', 'vaso': 'Vaso'
}

# --- Funciones de las secciones ---
def predict_demand_section():
    st.title("Predicción de Demanda")
    st.write("Aquí va la lógica de predicción de demanda.")  # Simplificado

def file_analysis_section():
    st.title("Análisis de Archivos")
    st.write("Aquí va la lógica de análisis de archivos.")  # Simplificado

def image_analysis_section():
    st.title("Análisis de Imágenes")
    st.write("Aquí va la lógica de análisis de imágenes.")  # Simplificado

def settings_section():
    st.title("Configuración")
    st.write("Aquí va la configuración del sistema.")  # Simplificado

# --- SELECCIÓN DEL TIPO DE NEGOCIO ---
if 'business_type' not in st.session_state:
    st.session_state.business_type = None

business_options = {
    "Restaurante": "🍽️ Soluciones para la gestión culinaria y de clientes.",
    "Clínica": "🏥 Optimización de procesos sanitarios y atención al paciente."
}

if st.session_state.business_type is None:
    st.title("Bienvenido a la Plataforma de IA Corporativa")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Soy un Restaurante"):
            st.session_state.business_type = "Restaurante"
            st.rerun()
    with col2:
        if st.button("Soy una Clínica"):
            st.session_state.business_type = "Clínica"
            st.rerun()
else:
    st.sidebar.title(f"Tipo de Negocio: {st.session_state.business_type}")

    if st.sidebar.button("Cambiar tipo de negocio"):
        st.session_state.business_type = None
        st.rerun()

    with st.sidebar:
        selected = option_menu(
            menu_title="Módulos de IA",
            options=["Predicción Demanda", "Análisis Archivos", "Análisis de Imágenes", "Configuración"],
            icons=["bar-chart-line", "file-earmark-text", "image", "gear"],
            default_index=0
        )

    if selected == "Predicción Demanda":
        predict_demand_section()
    elif selected == "Análisis Archivos":
        file_analysis_section()
    elif selected == "Análisis de Imágenes":
        image_analysis_section()
    elif selected == "Configuración":
        settings_section()
