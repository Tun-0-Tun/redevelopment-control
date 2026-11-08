"""Модели Claude по API (нужен ключ ANTHROPIC_API_KEY или `ant auth login`, пакет `anthropic`).

verify_changes: CNN нашла кандидатов -> Claude смотрит кроп «до | после» и подтверждает или отбрасывает каждого (гибрид).
detect_changes_vlm: Claude сам ищет изменения по двум картинкам, без CNN.
Расход: один запрос на пару, около 35 тыс. токенов по замерам исследования. Модели: claude-sonnet-5-5 (по умолчанию), claude-opus-5-5, claude-fable-5-1.
"""
import base64
import json
import re

import cv2
import numpy as np

from .align import align_ref_to_cur
from .changes import CLASSES
from .imageio import to_gray

DEFAULT_MODEL = "claude-sonnet-5-5"
_RULES = """Сканы могут отличаться растрированием: поворот, масштаб, сдвиг листа, шум, JPEG, 1-бит, низкое разрешение, перекос, перспектива. Это НЕ изменения.
Реальное изменение геометрии плана бывает четырёх типов:
 1 = removed_partition: перегородка есть в A, нет в B
 2 = added_partition: перегородки нет в A, есть в B
 3 = plumbing_moved: сантехнический прибор перенесён
 4 = doorway_moved: дверной проём перенесён в другое место стены
Если не уверен, не выдумывай изменение. Ответ: только JSON, без текста вокруг."""


def _client(client):
    if client is not None: return client
    try:
        import anthropic
    except ImportError as e:
        raise RuntimeError("для режима Claude нужен пакет: pip install anthropic") from e
    return anthropic.Anthropic()  # ключ берётся из ANTHROPIC_API_KEY или из профиля `ant auth login`


def _img(im):
    data = base64.standard_b64encode(cv2.imencode(".png", im)[1].tobytes()).decode()
    return {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}}


def _ask(client, model, content, max_tokens=2000):
    r = client.messages.create(model=model, max_tokens=max_tokens, messages=[{"role": "user", "content": content}])
    if r.stop_reason == "refusal":
        raise RuntimeError(f"модель отказалась отвечать: {getattr(r, 'stop_details', None)}")
    text = "".join(b.text for b in r.content if b.type == "text")
    m = re.search(r"\{.*\}", text, re.S)
    if not m: raise RuntimeError(f"ответ без JSON: {text[:200]}")
    return json.loads(m.group(0))


def _crop(a, cur, bbox):
    h, w = cur.shape
    x0, y0, x1, y1 = bbox
    m = max(25, 0.6 * max(x1 - x0, y1 - y0))
    X0, Y0, X1, Y1 = int(max(0, x0 - m)), int(max(0, y0 - m)), int(min(w, x1 + m)), int(min(h, y1 + m))
    f = 3 if (X1 - X0) < 250 else 2
    tile = lambda im: cv2.resize(im[Y0:Y1, X0:X1], None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
    A, B = tile(a), tile(cur)
    return np.hstack([A, np.zeros((A.shape[0], 8), np.uint8), B])


def verify_changes(before, after, result, model=DEFAULT_MODEL, client=None):
    """result: ответ detect_changes. Возвращает копию, где остались только подтверждённые кандидаты (класс берётся у Claude)."""
    cands = result["changes"][:6]
    if not cands: return result
    ref, cur = to_gray(before), to_gray(after)
    a, _ = align_ref_to_cur(ref, cur)
    content = [{"type": "text", "text": "Автоматический детектор отметил места возможных изменений плана. Каждый кроп: слева старая версия A, справа новая B. Часто это ложное срабатывание из-за шума или растра.\n" + _RULES +
                '\nФормат: {"candidates":[{"i":0,"real":true,"class":2,"confidence":0.8},{"i":1,"real":false}]}'}]
    for i, c in enumerate(cands):
        content += [{"type": "text", "text": f"Кандидат {i}:"}, _img(_crop(a, cur, c["bbox"]))]
    verdicts = {v["i"]: v for v in _ask(_client(client), model, content).get("candidates", []) if isinstance(v.get("i"), int)}
    out = []
    for i, c in enumerate(cands):
        v = verdicts.get(i)
        if v and v.get("real") and v.get("class") in CLASSES:
            cid = int(v["class"])
            out.append({**c, "class_id": cid, "class": CLASSES[cid][0], "label_ru": CLASSES[cid][1],
                        "score": round(float(v.get("confidence", c["score"])), 3), "verified_by": model, "detector_class": c["class"]})
    return {**result, "has_changes": bool(out), "changes": out}


def detect_changes_vlm(before, after, model=DEFAULT_MODEL, client=None):
    """Claude сам ищет изменения. Рамки приблизительные: модели хуже локализуют, чем CNN."""
    ref, cur = to_gray(before), to_gray(after)
    h, w = cur.shape
    content = [{"type": "text", "text": "Сравни два скана поэтажного плана: A (старая версия) и B (новая версия).\n" + _RULES +
                '\nДля каждого изменения дай bbox на картинке B в нормированных координатах всего файла B [x0,y0,x1,y1] (числа 0..1), с запасом 10-20 px.'
                '\nФормат: {"changes":[{"class":1,"bbox":[0.1,0.2,0.3,0.25],"confidence":0.8}]}. Если изменений нет: {"changes":[]}'},
               {"type": "text", "text": "A:"}, _img(ref), {"type": "text", "text": "B:"}, _img(cur)]
    items = []
    for c in _ask(_client(client), model, content).get("changes", []):
        if c.get("class") in CLASSES:
            cid = int(c["class"]); x0, y0, x1, y1 = c["bbox"]
            items.append({"class_id": cid, "class": CLASSES[cid][0], "label_ru": CLASSES[cid][1],
                          "bbox": [round(x0 * w, 1), round(y0 * h, 1), round(x1 * w, 1), round(y1 * h, 1)],
                          "score": round(float(c.get("confidence", 0.5)), 3), "verified_by": model})
    return {"image_size": [w, h], "aligned": None, "has_changes": bool(items), "changes": items}
