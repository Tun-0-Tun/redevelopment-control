import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .imageio import to_gray
from .model import device, load

CLASSES = ["outer_wall", "partition", "window", "door", "sanitary"]
LABELS_RU = {"outer_wall": "внешняя стена", "partition": "перегородка", "window": "окно", "door": "дверь/проём", "sanitary": "сантехника"}
W0 = 1100


def parse_entities(plan, thr=0.5, min_px=40):
    """Разбирает один план на сущности. Координаты рамок — в пикселях входного изображения."""
    im = to_gray(plan)
    h0, w0 = im.shape
    k = W0 / w0
    im = cv2.resize(im, (W0, round(h0 * k)), interpolation=cv2.INTER_AREA)
    h, w = im.shape
    x = np.full(((h + 31) // 32 * 32, (w + 31) // 32 * 32), 250, np.uint8)
    x[:h, :w] = im
    dev = device()
    t = torch.from_numpy(x)[None, None].repeat(1, 3, 1, 1).to(dev)
    with torch.no_grad():
        p = F.softmax(load("entity_unet", 6)((t.float() / 255 - 0.45) / 0.225), 1)[0, :, :h, :w].cpu().numpy()
    items = []
    for c in range(1, 6):
        m = (p[c] > thr).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        # стены и перегородки — линии: режем маску на горизонтальные и вертикальные отрезки
        parts = [m] if c >= 3 else [cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((1, 22), np.uint8)),
                                    cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((22, 1), np.uint8))]
        for mm in parts:
            n, lab, st, _ = cv2.connectedComponentsWithStats(mm)
            for i in range(1, n):
                x0, y0, ww, hh, area = st[i]
                if area < min_px: continue
                name = CLASSES[c - 1]
                items.append({"class": name, "label_ru": LABELS_RU[name],
                              "bbox": [round(float(v) / k, 1) for v in (x0, y0, x0 + ww, y0 + hh)],
                              "score": round(float(p[c][lab == i].mean()), 3)})
    return {"image_size": [w0, h0], "entities": items}
