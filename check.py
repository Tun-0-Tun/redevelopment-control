"""Дымовая проверка: python check.py. Падает с assert, если модели сломаны."""
import json
from pathlib import Path

from redevelopment_control import detect_changes, parse_entities

E = Path(__file__).parent / "examples"
gt = json.load(open(E / "ground_truth.json"))


def hit(p, g):
    cx, cy = (p[0] + p[2]) / 2, (p[1] + p[3]) / 2
    return g[0] <= cx <= g[2] and g[1] <= cy <= g[3]


r = detect_changes(E / "changed_before.png", E / "changed_after.png")
for g in gt["changed"]:
    assert any(c["class_id"] == g["class_id"] and hit(c["bbox"], g["bbox"]) for c in r["changes"]), f"не найдено изменение {g}"
r0 = detect_changes(E / "unchanged_before.png", E / "unchanged_after.png")
assert not r0["has_changes"], f"ложная тревога: {r0['changes']}"
ent = parse_entities(E / "changed_before.png")["entities"]
kinds = {e["class"] for e in ent}
assert {"window", "door", "sanitary"} <= kinds, kinds
print("OK: изменений найдено", len(r["changes"]), "| ложных на чистой паре 0 | сущностей", len(ent))

# --- режимы Claude: проверяем разбор ответов на подставном клиенте (реальный API здесь не вызывается)
import json as _j
from types import SimpleNamespace as NS
from redevelopment_control import detect_changes_vlm, verify_changes


class FakeClient:
    def __init__(self, reply): self.messages = NS(create=lambda **kw: NS(stop_reason="end_turn", content=[NS(type="text", text=reply)]))


base = detect_changes(E / "changed_before.png", E / "changed_after.png")
ok = verify_changes(E / "changed_before.png", E / "changed_after.png", base,
                    client=FakeClient(_j.dumps({"candidates": [{"i": 0, "real": True, "class": 3, "confidence": 0.9}, {"i": 1, "real": False}]})))
assert len(ok["changes"]) == 1 and ok["changes"][0]["class_id"] == 3 and ok["changes"][0]["verified_by"], ok
v = detect_changes_vlm(E / "changed_before.png", E / "changed_after.png",
                       client=FakeClient('текст {"changes":[{"class":1,"bbox":[0.5,0.4,0.6,0.5],"confidence":0.7}]} хвост'))
assert v["changes"][0]["bbox"] == [550.0, 332.0, 660.0, 415.0], v
print("OK: режимы Claude (на подставном клиенте)")
