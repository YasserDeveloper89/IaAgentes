import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression

def load_and_validate_sales(file) -> pd.DataFrame:
    try:
        df = pd.read_csv(file)
    except Exception as e:
        raise ValueError("Error al cargar archivo CSV. Revisa el formato.") from e

    # Validar columnas mínimas
    required_cols = {'fecha', 'ventas'}
    if not required_cols.issubset(df.columns.str.lower()):
        raise ValueError(f"El archivo debe contener las columnas: {required_cols}")

    # Normalizar nombres columnas a minúsculas
    df.columns = df.columns.str.lower()
    df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce')
    if df['fecha'].isnull().any():
        raise ValueError("Columna 'fecha' contiene valores inválidos o nulos.")
    if not np.issubdtype(df['ventas'].dtype, np.number):
        raise ValueError("Columna 'ventas' debe ser numérica.")

    return df.sort_values('fecha').reset_index(drop=True)

def predict_demand(df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, object]:
    """
    Modelo demo: regresión lineal simple para predecir demanda
    Retorna dataframe con demanda histórica + predicción y gráfico plotly
    """

    # Preparar datos para modelo
    df['day_index'] = (df['fecha'] - df['fecha'].min()).dt.days
    X = df[['day_index']]
    y = df['ventas']

    model = LinearRegression()
    model.fit(X, y)

    # Predecir para próximos 7 días
    last_day = df['day_index'].max()
    future_days = pd.DataFrame({'day_index': range(last_day + 1, last_day + 8)})
    future_dates = pd.date_range(df['fecha'].max() + pd.Timedelta(days=1), periods=7)

    preds = model.predict(future_days)
    preds = np.maximum(preds, 0)  # No negativas

    pred_df = pd.DataFrame({
        'fecha': future_dates,
        'ventas_predichas': preds.round(2)
    })

    result_df = pd.concat([
        df[['fecha', 'ventas']],
        pred_df.rename(columns={'ventas_predichas': 'ventas'})
    ], ignore_index=True)

    # Gráfico
    fig = px.line(result_df, x='fecha', y='ventas', title='Demanda Histórica y Predicción (7 días)', markers=True)
    fig.update_layout(hovermode="x unified")

    return result_df, fig
