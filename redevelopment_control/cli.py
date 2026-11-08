import argparse
import json
import sys

import cv2

from . import detect_changes, detect_changes_vlm, draw, parse_entities, verify_changes


def main():
    ap = argparse.ArgumentParser(prog="redevelopment_control", description="Сравнение планов и разбор сущностей")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("changes", help="найти перепланировку между двумя планами")
    c.add_argument("before"); c.add_argument("after")
    c.add_argument("--verify", action="store_true", help="проверить найденное моделью Claude (нужен ключ API)")
    c.add_argument("--vlm", action="store_true", help="искать изменения только моделью Claude, без нашей сети (нужен ключ API)")
    c.add_argument("--model", default="claude-sonnet-5-5", help="модель Claude для --verify / --vlm")
    e = sub.add_parser("entities", help="найти стены, окна, двери, сантехнику на одном плане")
    e.add_argument("plan")
    for p in (c, e):
        p.add_argument("--out", help="сохранить JSON в файл (по умолчанию в stdout)")
        p.add_argument("--overlay", help="сохранить картинку с рамками")
    a = ap.parse_args()
    if a.cmd == "changes":
        res = detect_changes_vlm(a.before, a.after, a.model) if a.vlm else detect_changes(a.before, a.after)
        if a.verify and not a.vlm: res = verify_changes(a.before, a.after, res, a.model)
        base, items, key = a.after, res["changes"], "class"
    else:
        res = parse_entities(a.plan); base, items, key = a.plan, res["entities"], "class"
    text = json.dumps(res, ensure_ascii=False, indent=1)
    open(a.out, "w").write(text) if a.out else sys.stdout.write(text + "\n")
    if a.overlay: cv2.imwrite(a.overlay, draw(base, items, key))
