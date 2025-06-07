import os
import yaml
import streamlit as st

# --- Cargar configuración ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config.yaml")

try:
    with open(CONFIG_PATH, "r") as f:
        CONFIG = yaml.safe_load(f)
except Exception as e:
    st.error(f"No se pudo cargar la configuración: {e}")
    st.stop()

# --- Aplicar estilos desde configuración ---
primary_color = CONFIG.get('ui', {}).get('primary_color', '#1f77b4')
accent_color = CONFIG.get('ui', {}).get('accent_color', '#ff7f0e')
font_family = CONFIG.get('ui', {}).get('font_family', 'Arial, sans-serif')
secondary_bg = CONFIG.get('ui', {}).get('secondaryBackgroundColor', '#f0f2f6')

st.markdown(f"""
    <style>
    body {{
        background-color: {secondary_bg};
        font-family: {font_family};
    }}
    .css-1d391kg {{
        background-color: {primary_color};
    }}
    .stButton>button {{
        background-color: {accent_color};
        color: white;
        border-radius: 8px;
        padding: 8px 16px;
    }}
    </style>
""", unsafe_allow_html=True)

# --- Título ---
st.title("Aplicación AI Agents + Video Analytics")
