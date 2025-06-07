# app/video_analytics.py
from typing import Tuple, Dict
from PIL import Image, ImageDraw
import numpy as np
import cv2
import logging

logger = logging.getLogger(__name__)

class VideoAnalysisError(Exception):
    pass

def detect_people(image_np: np.ndarray, config: Dict) -> int:
    """
    Método simple para detectar personas basado en contornos.
    Requiere imagen en formato numpy RGB o BGR.
    """
    try:
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        _, thresh = cv2.threshold(blurred, config.get("threshold", 60), 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filtrar contornos por tamaño mínimo
        min_area = config.get("min_contour_area", 500)
        person_count = sum(1 for cnt in contours if cv2.contourArea(cnt) > min_area)

        logger.info(f"Personas detectadas (contornos > {min_area}): {person_count}")
        return person_count

    except Exception as e:
        logger.error(f"Error en detect_people: {e}", exc_info=True)
        raise VideoAnalysisError("Error procesando la imagen para detección")

def analyze_occupancy(image_file, config: Dict) -> Tuple[Image.Image, int]:
    """
    Recibe archivo de imagen, analiza ocupación (personas).
    Retorna imagen anotada (PIL.Image) y cantidad detectada.
    """

    try:
        image = Image.open(image_file).convert("RGB")
        image_np = np.array(image)[:, :, ::-1].copy()  # PIL RGB -> OpenCV BGR

        count = detect_people(image_np, config)

        # Anotar imagen con número detectado
        draw = ImageDraw.Draw(image)
        text = f"Personas detectadas: {count}"
        draw.text((10, 10), text, fill="red")

        return image, count

    except Exception as ex:
        logger.error(f"Error en analyze_occupancy: {ex}", exc_info=True)
        raise VideoAnalysisError("Error analizando la imagen")
