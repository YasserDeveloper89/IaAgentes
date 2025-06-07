# app/ai_agent.py
from typing import Dict
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class DataValidationError(Exception):
    pass

def validate_input_data(df: pd.DataFrame) -> None:
    required_cols = {"producto", "cantidad", "fecha"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise DataValidationError(f"Faltan columnas obligatorias: {missing}")
    if df.empty:
        raise DataValidationError("El DataFrame está vacío")
    if not pd.api.types.is_numeric_dtype(df["cantidad"]):
        raise DataValidationError("La columna 'cantidad' debe ser numérica")
    # Validar formato fecha
    try:
        pd.to_datetime(df["fecha"])
    except Exception:
        raise DataValidationError("La columna 'fecha' tiene formato inválido")

def predict_demand(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """
    Recibe dataframe con ventas históricas y parámetros de configuración.
    Retorna dataframe con demanda estimada por insumo.
    """

    try:
        validate_input_data(df)
        logger.info("Datos validados correctamente para predicción de demanda")

        # Preprocesamiento básico
        df["fecha"] = pd.to_datetime(df["fecha"])
        df_grouped = (
            df.groupby(["producto"])
            .agg({"cantidad": "sum"})
            .rename(columns={"cantidad": "cantidad_total"})
            .reset_index()
        )

        # Parámetros del modelo (ejemplo simple)
        factor_crecimiento = config.get("growth_factor", 1.1)
        ajuste_temporal = config.get("temporal_adjustment", 1.0)

        # Predicción simple: sumar crecimiento al total histórico
        df_grouped["cantidad_estimda"] = (
            df_grouped["cantidad_total"] * factor_crecimiento * ajuste_temporal
        )

        # Round y tipo de dato
        df_grouped["cantidad_estimda"] = df_grouped["cantidad_estimda"].round(2)

        logger.info("Predicción de demanda calculada exitosamente")
        return df_grouped[["producto", "cantidad_estimda"]].rename(
            columns={"producto": "insumo"}
        )

    except DataValidationError as ve:
        logger.error(f"Error de validación: {ve}")
        raise
    except Exception as ex:
        logger.error(f"Error en predict_demand: {ex}", exc_info=True)
        raise RuntimeError("Error inesperado en predicción de demanda") from ex
