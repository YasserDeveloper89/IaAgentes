import streamlit as st
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def predict_demand_ui():
    st.header("Predicción de Demanda")

    uploaded_file = st.file_uploader("Carga tu archivo CSV con columnas: fecha, producto, cantidad", type=["csv"])

    if uploaded_file:
        try:
            data = pd.read_csv(uploaded_file, parse_dates=["fecha"])
        except Exception as e:
            st.error(f"Error leyendo el archivo CSV: {e}")
            return
        
        if not set(["fecha", "producto", "cantidad"]).issubset(data.columns):
            st.error("El CSV debe contener las columnas: 'fecha', 'producto', 'cantidad'")
            return
        
        productos = data["producto"].unique()
        producto_seleccionado = st.selectbox("Selecciona el producto para la predicción", productos)

        df_prod = data[data["producto"] == producto_seleccionado].copy()
        df_prod = df_prod.sort_values("fecha")

        st.markdown(f"### Datos históricos para *{producto_seleccionado}*")
        st.dataframe(df_prod.style.format({"cantidad": "{:,.0f}"}))

        # Modelo simple de Holt-Winters para predicción (puedes mejorar este modelo luego)
        try:
            model = ExponentialSmoothing(df_prod["cantidad"], trend="add", seasonal=None).fit()
            pred_len = st.slider("Días a predecir", 1, 30, 7)
            pred = model.forecast(pred_len)

            pred_df = pd.DataFrame({
                "fecha": pd.date_range(start=df_prod["fecha"].max() + pd.Timedelta(days=1), periods=pred_len),
                "Cantidad Predicha": pred.round().astype(int)
            })

            st.markdown("### Predicción de demanda futura")
            st.dataframe(pred_df.style.format({"Cantidad Predicha": "{:,.0f}"}))

            # Gráfico combinado
            fig, ax = plt.subplots(figsize=(10,5))
            sns.lineplot(x="fecha", y="cantidad", data=df_prod, marker="o", label="Histórico", ax=ax)
            sns.lineplot(x="fecha", y="Cantidad Predicha", data=pred_df, marker="o", label="Predicción", ax=ax)
            ax.set_title(f"Demanda histórica y predicha para {producto_seleccionado}")
            ax.set_xlabel("Fecha")
            ax.set_ylabel("Cantidad")
            ax.legend()
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Error en el modelo de predicción: {e}")
