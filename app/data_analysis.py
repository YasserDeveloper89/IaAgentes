import pandas as pd
import plotly.express as px

def analyze_uploaded_file(file) -> tuple[pd.DataFrame, object]:
    """
    Carga y análisis básico: resumen + gráfico distribución de una columna numérica (primera que encuentre).
    """

    try:
        df = pd.read_csv(file)
    except Exception as e:
        raise ValueError("Error al cargar archivo CSV.") from e

    # Resumen básico
    summary = df.describe(include='all').transpose()

    # Buscar primera columna numérica para gráfico
    num_cols = df.select_dtypes(include='number').columns
    if len(num_cols) == 0:
        raise ValueError("El archivo no contiene columnas numéricas para análisis gráfico.")

    fig = px.histogram(df, x=num_cols[0], nbins=30, title=f'Distribución de {num_cols[0]}')

    return summary, fig
