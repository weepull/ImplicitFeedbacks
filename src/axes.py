"""Dynamic attribute axes over CLIP space.

An axis is a contrast between two phrases (or two lists of paraphrases), e.g.
"photorealistic photograph" vs "digital illustration". Nothing is a fixed class:
any phrase pair becomes an axis at runtime, and every image gets a continuous score.

  score_A(image) = e_image . (mean(pos phrase embs) - mean(neg phrase embs))

Axes added at runtime persist in data/axes.json. Phrases can be given as
"a | b | c" to supply paraphrases; otherwise template variants are generated.
Reliability check: score images with each paraphrase separately; if the variants
rank images differently (low correlation), the axis depends on wording and is flagged.
"""
from __future__ import annotations

import json
import os

import numpy as np

AXES_PATH = "data/axes.json"
TEMPLATES = ("{}", "an image of {}", "a picture showing {}")
RELIABLE_CORR = 0.7

STARTER = {
    "photoreal vs illustrated": ("a photorealistic photograph", "a digital illustration"),
    "anime vs western": ("anime style artwork", "western comic or painting style"),
    "dark vs bright": ("a dark, moody, low-key image", "a bright, airy, high-key image"),
    "saturated vs muted": ("vivid saturated colours", "muted desaturated colours"),
    "dramatic vs flat light": ("dramatic high-contrast lighting", "flat even lighting"),
    "busy vs minimal": ("a crowded busy detailed frame", "a minimal clean empty frame"),
    "close-up vs wide": ("a tight close-up shot", "a wide establishing shot"),
    "low vs eye-level angle": ("a low angle shot looking up", "an eye level shot"),
    "action vs calm": ("a dynamic action scene with motion", "a calm still peaceful scene"),
    "cute vs serious": ("cute whimsical playful", "serious gritty realistic"),
    "fantasy vs everyday": ("an epic fantasy or sci-fi world", "an ordinary everyday scene"),
    "people vs no people": ("a portrait of a person", "a scene with no people"),
    "painterly vs crisp": ("painterly brush strokes texture", "a clean crisp sharp render"),
}


def _variants(text: str) -> list[str]:
    parts = [p.strip() for p in text.split("|") if p.strip()]
    if len(parts) > 1:
        return parts
    return [t.format(parts[0]) for t in TEMPLATES]


class AxisRegistry:
    def __init__(self, clip, image_emb: np.ndarray, path: str = AXES_PATH, sample: int = 4000):
        self.clip = clip
        self.path = path
        rng = np.random.default_rng(0)
        self.sample = image_emb[rng.choice(len(image_emb), min(sample, len(image_emb)), replace=False)]
        self.axes: dict[str, tuple[str, str]] = dict(STARTER)
        if os.path.exists(path):
            self.axes.update({k: tuple(v) for k, v in json.load(open(path)).items()})
        self._cache: dict[str, dict] = {}

    def add(self, name: str, pos: str, neg: str, persist: bool = True) -> dict:
        self.axes[name] = (pos, neg)
        self._cache.pop(name, None)
        if persist:
            custom = {k: v for k, v in self.axes.items() if k not in STARTER or STARTER[k] != v}
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            json.dump(custom, open(self.path, "w"), indent=1)
        return self.get(name)

    def get(self, name: str) -> dict:
        """-> {direction (768,), std over images, mean, reliability, variants}."""
        if name in self._cache:
            return self._cache[name]
        pos, neg = self.axes[name]
        pv, nv = _variants(pos), _variants(neg)
        P, N = self.clip.embed_texts(pv), self.clip.embed_texts(nv)
        direction = P.mean(0) - N.mean(0)
        # reliability: correlation of image scores across paraphrase variants
        k = min(len(pv), len(nv))
        per = np.stack([self.sample @ (P[i] - N[i]) for i in range(k)])
        corr = np.corrcoef(per)[np.triu_indices(k, 1)].mean() if k > 1 else np.nan
        s = self.sample @ direction
        out = dict(name=name, pos=pos, neg=neg, direction=direction, mean=float(s.mean()),
                   std=float(s.std()), reliability=float(corr), reliable=bool(corr >= RELIABLE_CORR))
        self._cache[name] = out
        return out

    def z(self, name: str, emb: np.ndarray) -> np.ndarray:
        """Standardised position of each image on the axis (0 = typical, +1 = 1 std toward pos)."""
        a = self.get(name)
        return (emb @ a["direction"] - a["mean"]) / a["std"]

    def names(self):
        return list(self.axes)
