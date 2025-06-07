import pandas as pd
import streamlit as st

def analyze_uploaded_file(uploaded_file):
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.success("Archivo cargado correctamente ✅")
        st.dataframe(df.head())

        st.subheader("📊 Resumen estadístico")
        st.write(df.describe())

        st.subheader("📈 Gráficos automáticos")
        numeric_cols = df.select_dtypes(include='number').columns
        if len(numeric_cols) >= 2:
            st.line_chart(df[numeric_cols])
        else:
            st.warning("No hay suficientes columnas numéricas para graficar.")

    except Exception as e:
        st.error(f"Ocurrió un error al analizar el archivo: {e}")
