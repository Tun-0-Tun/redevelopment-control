import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .align import align_ref_to_cur
from .imageio import to_gray
from .model import device, load

W, H = 512, 384
CLASSES = {1: ("removed_partition", "демонтаж перегородки"), 2: ("added_partition", "новая перегородка"),
           3: ("plumbing_moved", "перенос сантехники"), 4: ("doorway_moved", "перенос проёма")}
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def detect_changes(before, after, thr=0.7, min_px=25):
    """Сравнивает два плана (путь | bytes | массив). Координаты рамок — в пикселях изображения `after`."""
    ref, cur = to_gray(before), to_gray(after)
    h0, w0 = cur.shape
    aligned, ok = align_ref_to_cur(ref, cur)
    c = cv2.resize(cur, (W, H), interpolation=cv2.INTER_AREA)
    a = cv2.resize(aligned, (W, H), interpolation=cv2.INTER_AREA)
    x = np.stack([c, a, np.abs(c.astype(int) - a.astype(int)).astype(np.uint8)], -1)
    dev = device()
    t = torch.from_numpy(x).permute(2, 0, 1)[None].contiguous().to(dev)
    with torch.no_grad():
        p = F.softmax(load("change_unet", 5)(((t.float() / 255.0) - _MEAN.to(dev)) / _STD.to(dev)), 1)[0].cpu().numpy()
    fg = cv2.morphologyEx(((1 - p[0]) > thr).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(fg)
    sx, sy = w0 / W, h0 / H
    items = []
    for i in range(1, n):
        x0, y0, w, h, area = st[i]
        if area < min_px: continue
        m = lab == i
        cid = int(1 + p[1:][:, m].mean(1).argmax())
        items.append({"class_id": cid, "class": CLASSES[cid][0], "label_ru": CLASSES[cid][1],
                      "bbox": [round(float(v), 1) for v in (x0 * sx, y0 * sy, (x0 + w) * sx, (y0 + h) * sy)],
                      "score": round(float((1 - p[0])[m].mean()), 3)})
    items.sort(key=lambda d: -d["score"])
    return {"image_size": [w0, h0], "aligned": bool(ok), "has_changes": bool(items), "changes": items}
