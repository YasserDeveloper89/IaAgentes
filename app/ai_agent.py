import pandas as pd
import numpy as np

def predict_demand(file, config):
    """
    Recibe archivo CSV con ventas históricas y devuelve dataframe con predicción de demanda para próximas fechas.
    Espera columnas: fecha (YYYY-MM-DD), producto, cantidad_vendida
    """
    df = pd.read_csv(file)
    
    # Validación básica
    if not all(col in df.columns for col in ['fecha', 'producto', 'cantidad_vendida']):
        raise ValueError("El archivo debe contener columnas: fecha, producto, cantidad_vendida")
    
    # Convertir fecha a datetime
    df['fecha'] = pd.to_datetime(df['fecha'])
    
    # Agrupar ventas diarias por producto
    daily_sales = df.groupby(['fecha', 'producto'])['cantidad_vendida'].sum().reset_index()
    
    # Predicción sencilla: media móvil + crecimiento según config
    growth_factor = config.get('ai_agent', {}).get('growth_factor', 1.1)
    
    # Ultima fecha
    last_date = daily_sales['fecha'].max()
    
    # Generar predicciones para 7 días siguientes para cada producto
    products = daily_sales['producto'].unique()
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=7)
    
    predictions = []
    for product in products:
        prod_sales = daily_sales[daily_sales['producto'] == product].sort_values('fecha')
        # Media movil simple de los últimos 7 días
        rolling_avg = prod_sales['cantidad_vendida'].rolling(window=7, min_periods=1).mean().iloc[-1]
        
        for date in future_dates:
            predicted_qty = rolling_avg * growth_factor
            predictions.append({
                'fecha': date.strftime('%Y-%m-%d'),
                'producto': product,
                'cantidad_predicha': round(predicted_qty, 2)
            })
    
    pred_df = pd.DataFrame(predictions)
    return pred_df
