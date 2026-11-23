"""Measured image layers -> a few interpretable numbers per image.

Unlike CLIP axes (which depend on wording), these are direct measurements.
  --part color : brightness, contrast, saturation, warmth, colourfulness, palette entropy,
                 edge density, sharpness. CPU only. (Faces dropped: OpenCV 5 removed Haar
                 cascades, and CLIP's people axis already passes the paraphrase test.)
  --part depth : Depth Anything V2 (small): subject distance, depth range, foreground
                 fraction & centroid (composition), vertical depth gradient (angle proxy). GPU.
Writes data/measured_<part>.parquet keyed by uid.

  python src/measured.py --part color
  python src/measured.py --part depth
"""
from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
import pandas as pd

SIZE = 448


def _load(uid):
    im = cv2.imread(f"data/images/{uid}.jpg", cv2.IMREAD_REDUCED_COLOR_2)
    if im is None:
        return None
    h, w = im.shape[:2]
    s = SIZE / max(h, w)
    return cv2.resize(im, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)


def color_features(uid):
    im = _load(uid)
    if im is None:
        return None
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV).astype(np.float32)
    b, g, r = [im[..., i].astype(np.float32) for i in range(3)]
    rg, yb = r - g, 0.5 * (r + g) - b
    colourful = np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
    sat_mask = hsv[..., 1] > 40
    hue_hist = np.histogram(hsv[..., 0][sat_mask], bins=18, range=(0, 180))[0] + 1e-9
    hue_p = hue_hist / hue_hist.sum()
    edges = cv2.Canny(gray.astype(np.uint8), 100, 200)
    return dict(
        uid=uid,
        m_brightness=gray.mean() / 255,
        m_contrast=gray.std() / 255,
        m_dark_frac=(gray < 50).mean(),
        m_saturation=hsv[..., 1].mean() / 255,
        m_warmth=(r.mean() - b.mean()) / 255,
        m_colourfulness=colourful / 100,
        m_palette_entropy=float(-(hue_p * np.log(hue_p)).sum()) if sat_mask.any() else 0.0,
        m_edge_density=(edges > 0).mean(),
        m_sharpness=float(np.log1p(cv2.Laplacian(gray, cv2.CV_32F).var())),
    )


def depth_features(depth: np.ndarray) -> dict:
    """depth: relative inverse depth (larger = closer), HxW."""
    d = (depth - depth.min()) / (np.ptp(depth) + 1e-6)
    h, w = d.shape
    # foreground = near side of an automatic (Otsu) split, so a close-up subject can fill most of the frame
    thr, _ = cv2.threshold((d * 255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    fg = d * 255 > thr
    ys, xs = np.nonzero(fg)
    rows = d.mean(1)
    return dict(
        d_near=float(np.quantile(d, 0.95)),
        d_median=float(np.median(d)),
        d_range=float(np.quantile(d, 0.95) - np.quantile(d, 0.05)),
        d_fg_frac=float(fg.mean()),                          # big = subject fills frame (close-up)
        d_fg_cx=float(xs.mean() / w) if len(xs) else 0.5,    # subject horizontal position
        d_fg_cy=float(ys.mean() / h) if len(ys) else 0.5,
        d_fg_offcentre=float(np.hypot(xs.mean() / w - .5, ys.mean() / h - .5)) if len(xs) else 0.0,
        d_vgrad=float(np.polyfit(np.linspace(0, 1, h), rows, 1)[0]),  # >0: bottom closer (ground plane, eye/high angle)
        d_centre_minus_edge=float(d[h // 4:3 * h // 4, w // 4:3 * w // 4].mean() - d.mean()),
    )


def run_color(uids):
    with ProcessPoolExecutor(8) as ex:
        rows = [r for r in ex.map(color_features, uids, chunksize=64) if r]
    return pd.DataFrame(rows)


def run_depth(uids, batch=16):
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    name = "depth-anything/Depth-Anything-V2-Small-hf"
    proc = AutoImageProcessor.from_pretrained(name)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = AutoModelForDepthEstimation.from_pretrained(name).to(dev).eval()
    rows = []
    for i in range(0, len(uids), batch):
        ims = [Image.open(f"data/images/{u}.jpg").convert("RGB") for u in uids[i:i + batch]]
        for im in ims:
            im.draft("RGB", (SIZE, SIZE))
        with torch.no_grad():
            x = proc(images=ims, return_tensors="pt").to(dev)
            pred = model(**x).predicted_depth.float().cpu().numpy()
        rows += [dict(uid=u, **depth_features(p)) for u, p in zip(uids[i:i + batch], pred)]
        if (i // batch) % 100 == 0:
            print(f"{i + len(ims)}/{len(uids)}", flush=True)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["color", "depth"], required=True)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    z = np.load("data/embeddings.npz")
    uids = [u for u in z["uids"] if os.path.exists(f"data/images/{u}.jpg")][: a.limit]
    df = run_color(uids) if a.part == "color" else run_depth(uids)
    df.to_parquet(f"data/measured_{a.part}.parquet" if not a.limit else f"data/measured_{a.part}_sample.parquet")
    print(f"{len(df)} images written")
    print(df.describe().T[["mean", "std", "min", "max"]].round(3).to_string())
