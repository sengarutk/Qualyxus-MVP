from __future__ import annotations
import numpy as np
import cv2


def rle_decode(mask_rle: str, shape: tuple[int, int]) -> np.ndarray:
    """
    Decode RLE mask (Kaggle format) into binary mask.
    shape: (height, width)
    """
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0::2], s[1::2])]
    starts -= 1
    ends = starts + lengths

    h, w = shape
    img = np.zeros(h * w, dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    return img.reshape((w, h)).T  # column-major


def mask_to_polygons(mask: np.ndarray, min_area: int = 50) -> list[list[tuple[int, int]]]:
    mask_u8 = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polys = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        cnt = cnt.squeeze(axis=1)
        if len(cnt) < 3:
            continue
        polys.append([(int(x), int(y)) for x, y in cnt])
    return polys


def polygon_to_yolo_seg_line(poly: list[tuple[int, int]], w: int, h: int) -> str:
    coords = []
    for x, y in poly:
        coords.append(f"{x / w:.6f}")
        coords.append(f"{y / h:.6f}")
    return " ".join(coords)
