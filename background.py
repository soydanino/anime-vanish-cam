"""Recorte de la persona y foto del fondo vacío."""
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

CAPTURE_SECONDS = 4.0      # cuenta atrás para salir del encuadre
SAMPLE_SECONDS = 0.55      # se usan las imágenes del final de la cuenta atrás
MAX_PERSON_FRACTION = 0.01  # si más del 1 % del fondo parece una persona, se repite la captura


class PersonSegmenter:
    def __init__(self, models_dir: Path):
        self.seg = vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=str(models_dir / "selfie_segmenter_landscape.tflite")),
            running_mode=vision.RunningMode.VIDEO,
            output_confidence_masks=True,
            output_category_mask=False,
        ))
        self._ts = 0

    def mask(self, image: mp.Image, ts_ms: int, width: int, height: int) -> np.ndarray:
        """Máscara float32 0..1 de la persona (mismo ajuste que vanish_webcam)."""
        ts_ms = max(ts_ms, self._ts + 1)
        self._ts = ts_ms
        res = self.seg.segment_for_video(image, ts_ms)
        if not res.confidence_masks:
            return np.zeros((height, width), np.float32)
        raw = np.squeeze(res.confidence_masks[0].numpy_view()).astype(np.float32)
        m = cv2.resize(raw, (width, height), interpolation=cv2.INTER_LINEAR)
        m = cv2.GaussianBlur(m, (0, 0), sigmaX=2.2, sigmaY=2.2)
        return np.clip((m - 0.12) / 0.68, 0.0, 1.0)

    def close(self):
        self.seg.close()


class BackgroundCapture:
    """Cuenta atrás de 4 s; el fondo es la mediana de las imágenes de los últimos 0.55 s."""

    def __init__(self):
        self.background = None  # uint8 HxWx3
        self._start = None
        self._samples = []

    @property
    def capturing(self):
        return self._start is not None

    def start(self, now: float):
        self._start = now
        self._samples = []

    def seconds_left(self, now: float) -> float:
        return 0.0 if self._start is None else max(0.0, CAPTURE_SECONDS - (now - self._start))

    def feed(self, frame: np.ndarray, now: float):
        """Devuelve la imagen candidata cuando termina la cuenta atrás; si no, None."""
        if self._start is None:
            return None
        elapsed = now - self._start
        if elapsed >= CAPTURE_SECONDS - SAMPLE_SECONDS:
            self._samples.append(frame.copy())
        if elapsed < CAPTURE_SECONDS:
            return None
        stack = np.stack(self._samples[-18:]) if self._samples else frame[None]
        self._start, self._samples = None, []
        return np.median(stack, axis=0).astype(np.uint8)


def person_fraction(mask: np.ndarray) -> float:
    return float((mask > 0.5).mean())
