import cv2

from .imageio import to_gray

COL = {"removed_partition": (60, 60, 230), "added_partition": (60, 160, 40), "plumbing_moved": (200, 60, 200), "doorway_moved": (230, 140, 20),
       "outer_wall": (30, 60, 220), "partition": (230, 140, 30), "window": (0, 190, 210), "door": (60, 160, 50), "sanitary": (200, 50, 200)}


def draw(image, items, key="class"):
    """Рисует рамки из результата detect_changes (key='class', items=res['changes']) или parse_entities (items=res['entities'])."""
    im = cv2.cvtColor(to_gray(image), cv2.COLOR_GRAY2BGR)
    for it in items:
        x0, y0, x1, y1 = map(int, it["bbox"])
        col = COL.get(it[key], (0, 0, 0))
        cv2.rectangle(im, (x0, y0), (x1, y1), col, 2)
        cv2.putText(im, f"{it[key]} {it['score']:.2f}", (x0, max(10, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
    return im
