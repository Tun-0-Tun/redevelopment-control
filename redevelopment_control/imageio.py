from pathlib import Path

import cv2
import numpy as np


def to_gray(x):
    """Путь, bytes или numpy-массив (BGR/серый) -> серый uint8."""
    if isinstance(x, (str, Path)):
        im = cv2.imread(str(x), cv2.IMREAD_GRAYSCALE)
        if im is None: raise ValueError(f"не удалось прочитать изображение: {x}")
        return im
    if isinstance(x, (bytes, bytearray)):
        im = cv2.imdecode(np.frombuffer(x, np.uint8), cv2.IMREAD_GRAYSCALE)
        if im is None: raise ValueError("не удалось декодировать изображение")
        return im
    x = np.asarray(x)
    return cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) if x.ndim == 3 else x.astype(np.uint8)
