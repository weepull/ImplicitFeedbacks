# Taste Engine

Learns a person's visual taste from pairwise choices ("which of these two images do you prefer?"), predicts how likely they are to like a new AI-generated image with a calibrated confidence range, ranks candidates, and explains why ("+ photoreal, + close-up, − busy background").

Built and evaluated on [Pick-a-Pic v1](https://huggingface.co/datasets/pickapic-anonymous/pickapic_v1), a public dataset of AI-generated image pairs with real users' preferences.

## Results so far

Held-out test: each of 10 heavy Pick-a-Pic raters' *later* 30% of choices, predicted from their earlier ones (1,493 pairs).

| Method | Accuracy |
|---|---|
| Coin flip | 50.0% |
| LAION aesthetic ("prettiness") score | 47.2% |
| Prompt match only (CLIP, no training) | 65.0% |
| Population taste (other users) | 68.5% |
| **Personal taste + population prior** | **70.0%** |

- Confidence is well calibrated (expected calibration error 0.029).
- Among 3+ images for one prompt, the top pick is the user's favourite 38–40% of the time vs 21% by chance.
- Personal over population is +1.5 points, **not yet statistically certain**. Several follow-ups (low-rank personalisation, context conditioning) were tested and did not help. See `CLAUDE.md` → "Findings" and `docs/roadmap.md`.

## Setup

Requires Python 3.11. Developed on Apple Silicon (PyTorch MPS, MLX); CLIP and depth also run on CPU or CUDA, just slower. The Qwen VLM labeller (`vlm_label.py`) needs Apple Silicon.

```bash
python3.11 -m venv .venv            # or: uv venv -p 3.11 .venv
.venv/bin/pip install -r requirements.txt
```

## Rebuilding the data

`data/` is not in the repo (≈4 GB, regenerated). Pick-a-Pic image URLs are dead, so whole ~500 MB shards are streamed, embedded and deleted.

```bash
.venv/bin/python src/scan_meta.py                    # ~15 min: find heavy raters
.venv/bin/python src/build_dataset.py --users 10 --shards 80 --global-shards 12   # ~1 h at ~13 MB/s
.venv/bin/python src/build_dataset.py --merge
.venv/bin/python src/evaluate.py                     # trains + evaluates; required before the apps
.venv/bin/python src/measured.py --part color        # pixel features for the demo
```

## Running

```bash
.venv/bin/python src/demo_app.py                 # demo: http://localhost:8770
.venv/bin/python src/rate_app.py --user abc      # rate pairs to build your own profile: http://localhost:8765
```

`--user` is any anonymous name; ratings are stored locally in `data/feedback/<user>.jsonl` and never leave the machine.

## Layout

- `src/taste.py`: the taste model (Bayesian Bradley-Terry over CLIP features, population prior)
- `src/axes.py`, `src/conditional.py`: readable attribute axes defined by phrases at runtime, and "taste in a context"
- `src/measured.py`: pixel and depth layers; `src/vlm_label.py`, `src/check_axes.py`: checking attributes against labels
- `src/evaluate.py`, `src/verify.py`, `src/eval_lowrank.py`: every quality claim comes from these
- `reports/`: latest evaluation outputs
- `CLAUDE.md`: architecture, commands and findings in more depth
