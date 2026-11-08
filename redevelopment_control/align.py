import cv2
import numpy as np


def _ink(im, sigma):
    f = 1.0 - cv2.GaussianBlur(im.astype(np.float32) / 255.0, (0, 0), sigma)
    return (f - f.mean()) / (f.std() + 1e-6)


def align_ref_to_cur(ref, cur):
    """Подгоняет старый план под кадр нового: resize + ECC-гомография по пирамиде. Возвращает (картинка, сошлось ли)."""
    h, w = cur.shape
    r = cv2.resize(ref, (w, h), interpolation=cv2.INTER_AREA)
    warp = np.eye(3, dtype=np.float32)
    ok = True
    for scale, sigma in ((0.25, 1.5), (0.5, 2.0), (1.0, 2.5)):
        s = lambda im: cv2.resize(im, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        a, b = _ink(s(r), sigma), _ink(s(cur), sigma)
        S = np.diag([scale, scale, 1]).astype(np.float32)
        w_s = S @ warp @ np.linalg.inv(S)
        try:
            _, w_s = cv2.findTransformECC(b, a, w_s, cv2.MOTION_HOMOGRAPHY,
                                          (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5), None, 5)
            warp = np.linalg.inv(S) @ w_s @ S
        except cv2.error:
            ok = False
    out = cv2.warpPerspective(r, warp, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderValue=255)
    return out, ok
