import pandas as pd
import numpy as np
import plotly.express as px

def analyze_uploaded_file(file):
    """
    Análisis básico de archivo CSV:
    - Estadísticas por columnas numéricas
    - Conteo de valores faltantes
    - Detección simple de outliers (z-score)
    """
    df = pd.read_csv(file)
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    if len(numeric_cols) == 0:
        raise ValueError("El archivo no contiene columnas numéricas para análisis")
    
    analysis = {}
    for col in numeric_cols:
        col_data = df[col]
        stats = {
            'media': col_data.mean(),
            'mediana': col_data.median(),
            'desviacion_std': col_data.std(),
            'valores_faltantes': col_data.isna().sum(),
            'outliers': ((np.abs((col_data - col_data.mean()) / col_data.std()) > 3).sum())
        }
        analysis[col] = stats
    
    # Convertir a dataframe para mostrar en tabla
    report = pd.DataFrame(analysis).T.reset_index().rename(columns={'index': 'columna'})
    return report

def advanced_summary(file):
    """
    Genera un gráfico resumen interactivo usando Plotly.
    Ejemplo: Distribución y evolución temporal para la primera columna numérica.
    """
    df = pd.read_csv(file)
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    if len(numeric_cols) == 0:
        raise ValueError("El archivo no contiene columnas numéricas para análisis")
    
    # Tomamos la primera columna numérica para ejemplo
    col = numeric_cols[0]
    
    if 'fecha' in df.columns:
        df['fecha'] = pd.to_datetime(df['fecha'])
        fig = px.line(df, x='fecha', y=col, title=f"Evolución temporal de {col}")
    else:
        fig = px.histogram(df, x=col, nbins=30, title=f"Distribución de {col}")
    
    return fig
