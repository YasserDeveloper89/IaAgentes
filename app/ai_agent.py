import pandas as pd
import logging

# Configura logging básico
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def predict_demand(uploaded_file, config):
    """
    Función para predecir demanda basada en un archivo CSV con datos de ventas.
    Args:
        uploaded_file: archivo CSV subido por el usuario
        config: diccionario con configuración (puede usarse para ajustar parámetros)

    Returns:
        Un DataFrame con predicciones o resumen de datos (ejemplo simple aquí)

    Lanza:
        RuntimeError con mensaje y la excepción original si ocurre algún error.
    """
    try:
        # Leer el CSV
        df = pd.read_csv(uploaded_file)

        # Columnas esperadas en el CSV
        expected_columns = ['fecha', 'producto', 'cantidad_vendida', 'precio_unitario']
        missing_cols = [col for col in expected_columns if col not in df.columns]

        if missing_cols:
            error_msg = f"Faltan columnas requeridas en el archivo: {missing_cols}"
            logging.error(error_msg)
            raise ValueError(error_msg)

        # Validar que 'fecha' pueda convertirse a datetime
        try:
            df['fecha'] = pd.to_datetime(df['fecha'])
        except Exception as ex:
            logging.error(f"Error al convertir la columna 'fecha' a datetime: {ex}")
            raise ValueError("La columna 'fecha' debe tener formato de fecha válido") from ex

        # Aquí iría la lógica real de predicción.
        # Por simplicidad, hacemos un ejemplo: calcular demanda total por producto
        demanda_total = df.groupby('producto')['cantidad_vendida'].sum().reset_index()
        demanda_total.rename(columns={'cantidad_vendida': 'demanda_total'}, inplace=True)

        logging.info("Predicción de demanda calculada correctamente.")
        return demanda_total

    except Exception as ex:
        logging.error("Error en predict_demand", exc_info=True)
        raise RuntimeError("Error inesperado en predicción de demanda") from ex
