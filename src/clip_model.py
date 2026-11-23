"""CLIP wrapper: image/text embeddings (L2-normalised) + LAION aesthetic score.

ViT-L-14 (openai) is used rather than B-32 because the LAION aesthetic MLP is
trained on exactly that embedding space, giving us a free "pretty" baseline.
"""
from __future__ import annotations

import io
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import open_clip
import torch
from PIL import Image

MODEL_NAME = "ViT-L-14"
PRETRAINED = "openai"
EMB_DIM = 768
AESTHETIC_URL = (
    "https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/"
    "sac+logos+ava1-l14-linearMSE.pth"
)
AESTHETIC_PATH = "data/models/aesthetic_l14.pth"


def device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


class Clip:
    def __init__(self):
        self.dev = device()
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            MODEL_NAME, pretrained=PRETRAINED, device=self.dev
        )
        self.model.eval()
        self.dtype = torch.float16 if self.dev == "mps" else torch.float32
        self.model.to(self.dtype)  # fp16 on MPS: ~4.5x faster, cosine vs fp32 >= 0.9999
        self.tokenizer = open_clip.get_tokenizer(MODEL_NAME)
        self._pool = ThreadPoolExecutor(8)

    def _load(self, src) -> torch.Tensor:
        if isinstance(src, (bytes, bytearray)):
            img = Image.open(io.BytesIO(src))
        else:
            img = Image.open(src)
        img.draft("RGB", (448, 448))  # JPEG DCT downscale on decode: ~5x faster, no quality loss at 224px
        return self.preprocess(img.convert("RGB"))

    @torch.no_grad()
    def embed_images(self, sources, batch_size: int = 64) -> np.ndarray:
        """sources: list of file paths or raw jpeg bytes -> (n, 768) float32, L2-normalised."""
        out = []
        for i in range(0, len(sources), batch_size):
            batch = list(self._pool.map(self._load, sources[i : i + batch_size]))
            x = torch.stack(batch).to(self.dev, self.dtype)
            e = self.model.encode_image(x).float()
            out.append(torch.nn.functional.normalize(e, dim=-1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, EMB_DIM), np.float32)

    @torch.no_grad()
    def embed_texts(self, texts: list[str], batch_size: int = 256) -> np.ndarray:
        out = []
        for i in range(0, len(texts), batch_size):
            tok = self.tokenizer(texts[i : i + batch_size]).to(self.dev)
            e = self.model.encode_text(tok).float()
            out.append(torch.nn.functional.normalize(e, dim=-1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, EMB_DIM), np.float32)


class Aesthetic:
    """LAION improved-aesthetic-predictor (linear-MSE MLP on normalised L/14 embeddings)."""

    def __init__(self):
        if not os.path.exists(AESTHETIC_PATH):
            os.makedirs(os.path.dirname(AESTHETIC_PATH), exist_ok=True)
            torch.hub.download_url_to_file(AESTHETIC_URL, AESTHETIC_PATH)
        sd = torch.load(AESTHETIC_PATH, map_location="cpu")
        self.net = torch.nn.Sequential(
            torch.nn.Linear(768, 1024), torch.nn.Dropout(0.2),
            torch.nn.Linear(1024, 128), torch.nn.Dropout(0.2),
            torch.nn.Linear(128, 64), torch.nn.Dropout(0.1),
            torch.nn.Linear(64, 16), torch.nn.Linear(16, 1),
        )
        self.net.load_state_dict({k.replace("layers.", ""): v for k, v in sd.items()})
        self.net.eval()

    @torch.no_grad()
    def score(self, emb: np.ndarray) -> np.ndarray:
        return self.net(torch.from_numpy(emb.astype(np.float32))).squeeze(-1).numpy()
