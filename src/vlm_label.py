"""Label images with a free local vision-language model (Qwen2.5-VL-3B, 4-bit, MLX).

Purpose: ground truth to check whether CLIP axes measure what their names say,
and training labels for a small CLIP-based tagger on the axes CLIP is weak at.

  python src/vlm_label.py --n 20 --bench     # speed + sanity check
  python src/vlm_label.py --n 2000           # resumable; appends to data/labels/vlm.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import time

import numpy as np
from PIL import Image

MODEL = "mlx-community/Qwen2.5-VL-3B-Instruct-4bit"
OUT = "data/labels/vlm.jsonl"

# attribute -> allowed values. Mapped to CLIP axes in check_axes.py.
SCHEMA = {
    "style": ["photo", "illustration", "anime", "3d_render", "painting", "other"],
    "shot": ["close_up", "medium", "wide"],
    "angle": ["low", "eye_level", "high"],
    "lighting": ["dramatic", "flat"],
    "brightness": ["dark", "medium", "bright"],
    "clutter": ["busy", "minimal"],
    "energy": ["action", "calm"],
    "people": ["yes", "no"],
    "setting": ["fantasy", "everyday"],
    "mood": ["cute", "neutral", "serious"],
    "saturation": ["vivid", "muted"],
}
QUESTIONS = {
    "style": "rendering style",
    "shot": "camera distance to the main subject",
    "angle": "camera angle (low = looking up at subject, high = looking down)",
    "lighting": "dramatic high-contrast lighting with strong shadows, or flat even lighting",
    "brightness": "overall brightness",
    "clutter": "busy/cluttered frame or minimal/clean frame",
    "energy": "dynamic action/motion or calm/still",
    "people": "is a person or human-like character the main subject",
    "setting": "fantasy/sci-fi/imaginary or everyday/realistic subject",
    "mood": "cute/whimsical, neutral, or serious/gritty",
    "saturation": "vivid saturated colours or muted/desaturated colours",
}


def prompt_text():
    lines = [f'"{k}": one of {"|".join(v)}  ({QUESTIONS[k]})' for k, v in SCHEMA.items()]
    return ("Look at the image and answer each field with exactly one allowed value. "
            "Reply with a single JSON object and nothing else.\n" + "\n".join(lines))


def parse(text: str) -> dict:
    out = {}
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            d = json.loads(m.group(0))
            out = {k: str(d.get(k, "")).strip().lower().replace(" ", "_").replace("-", "_") for k in SCHEMA}
        except json.JSONDecodeError:
            pass
    if not out:  # regex fallback for slightly malformed JSON
        for k in SCHEMA:
            mm = re.search(rf'"{k}"\s*:\s*"([^"]+)"', text)
            out[k] = mm.group(1).strip().lower().replace(" ", "_") if mm else ""
    return {k: (v if v in SCHEMA[k] else None) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--size", type=int, default=448)
    a = ap.parse_args()

    from mlx_vlm import generate, load
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import load_config

    z = np.load("data/embeddings.npz")
    uids = [u for u in z["uids"] if os.path.exists(f"data/images/{u}.jpg")]
    rng = np.random.default_rng(7)
    order = [uids[i] for i in rng.permutation(len(uids))][: a.n]

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    done = set()
    if os.path.exists(OUT) and not a.bench:
        done = {json.loads(l)["uid"] for l in open(OUT)}
    todo = [u for u in order if u not in done]
    print(f"{len(todo)} images to label ({len(done)} already done)", flush=True)

    model, processor = load(MODEL)
    config = load_config(MODEL)
    prompt = apply_chat_template(processor, config, prompt_text(), num_images=1)
    tmp = tempfile.mkdtemp()
    t0, n_bad = time.time(), 0
    out = open(OUT if not a.bench else os.devnull, "a")
    for i, u in enumerate(todo):
        im = Image.open(f"data/images/{u}.jpg").convert("RGB")
        im.thumbnail((a.size, a.size))
        fp = os.path.join(tmp, "x.jpg")
        im.save(fp, quality=90)
        r = generate(model, processor, prompt, image=[fp], max_tokens=160, temperature=0.0, verbose=False)
        text = r.text if hasattr(r, "text") else str(r)
        lab = parse(text)
        n_bad += sum(v is None for v in lab.values())
        out.write(json.dumps({"uid": u, **lab}) + "\n"); out.flush()
        if a.bench and i < 3:
            print(text.strip()[:400], "\n->", lab, flush=True)
        if (i + 1) % 50 == 0 or (a.bench and i + 1 == len(todo)):
            el = time.time() - t0
            print(f"[{i + 1}/{len(todo)}] {el / (i + 1):.2f}s/img, invalid fields {n_bad / ((i + 1) * len(SCHEMA)):.1%}, "
                  f"eta {el / (i + 1) * (len(todo) - i - 1) / 60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
