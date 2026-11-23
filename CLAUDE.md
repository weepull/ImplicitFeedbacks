# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A personal visual-taste engine for AI-generated images: learn a person's taste from pairwise choices ("which of these two do you prefer?"), predict P(this person likes this image) with a calibrated interval, rank candidates, and explain why. The target user is the anonymous `abc`; until they have rated enough, method development uses 10 heavy raters from Pick-a-Pic v1 as stand-ins.

[day1-plan.md](day1-plan.md) was the original Day 1 spec (no training/reranker/UI). That scope has since been explicitly expanded by the user; the plan's probe and sanity-check ideas still underpin `axes.py` and `verify.py`. Parked work and its go/no-go criteria live in [docs/roadmap.md](docs/roadmap.md).

## Environment

Python 3.11 venv at `.venv` (created with `uv`; system python is 3.9 and lacks everything). Always run `.venv/bin/python -W ignore src/<script>.py` from the repo root: all paths are relative to it.

Apple Silicon (M5, 16GB). CLIP runs on MPS in **fp16** (4.5x faster than fp32, cosine ≥ 0.9999). Only one GPU-heavy job at a time is comfortable: the Qwen VLM, depth extraction and dataset embedding all compete for MPS/memory.

## Commands, in pipeline order

```bash
# data (already done; resumable). Image URLs in Pick-a-Pic are dead, so whole ~500MB shards are streamed.
.venv/bin/python src/scan_meta.py                      # metadata of all 387 shards -> data/meta.parquet (~15 min)
.venv/bin/python src/build_dataset.py --users 10 --shards 80 --global-shards 12   # stream + embed (~1 h)
.venv/bin/python src/build_dataset.py --merge          # -> data/embeddings.npz, data/pairs_ready.parquet

# evaluation (writes data/profiles/_population.npz, which every app needs)
.venv/bin/python src/evaluate.py                       # baselines vs population vs personal; reports/eval.json
.venv/bin/python src/eval_lowrank.py                   # low-rank experiment (lost; see findings)
.venv/bin/python src/conditional.py --eval             # does context conditioning help? (go/no-go for option D)
.venv/bin/python src/verify.py {splits|rank|readout|paraphrase|app|all}   # appends to reports/verify.txt

# attribute layers + checks
.venv/bin/python src/measured.py --part color          # pixel features, CPU, ~30 s
.venv/bin/python src/measured.py --part depth          # Depth Anything V2 small, GPU
.venv/bin/python src/vlm_label.py --n 2000             # Qwen2.5-VL-3B labels (~2 s/img), resumable
.venv/bin/python src/check_axes.py                     # AUC of CLIP / tagger / measured vs labels; reports/axes_check.csv

# apps
.venv/bin/python src/demo_app.py                       # demo, http://localhost:8770
.venv/bin/python src/rate_app.py --user abc          # rating + profile, http://localhost:8765
.venv/bin/python src/label_app.py --n 100              # blind human labelling, http://localhost:8766
.venv/bin/python src/rerank.py --user 3064 --prompt "..." --given "photorealistic" img1.jpg ...
.venv/bin/python src/conditional.py --user 3064 --given "a portrait" [--add "name=pos phrase=neg phrase"]
```

There is no test suite; `verify.py` is the closest thing, and every claim about model quality should come from `evaluate.py`/`verify.py` output, not from smoke tests.

## Architecture

**Data.** `data/pairs.parquet` is the *selection* (role = `target` for the 10 stand-in users, `global` for population pairs) and must not be overwritten; `data/pairs_ready.parquet` is the subset whose images were actually embedded, and is what everything reads. `data/embeddings.npz` holds L2-normalised CLIP ViT-L-14 (openai) image embeddings, caption embeddings, and LAION aesthetic scores (ViT-L-14 was chosen because that MLP is trained on its space). Images live in `data/images/<uid>.jpg`.

**Taste model (`taste.py`).** Bayesian Bradley-Terry: P(A ≻ B) = σ(w · (x_A − x_B)), x = [768-d image embedding, prompt alignment, aesthetic], features divided by `scale` (std of population pair diffs). Population weights are fit on `global` pairs; each user is a MAP fit with prior N(w_pop, I/λ) and a Laplace posterior. P(like) = probability of beating a random reference-pool image, averaged over posterior samples (alignment zeroed, i.e. taste only). `fit_map` supports `sample_weight` (conditioning) and `offset` (low-rank).

**Profiles (`user_profile.py`).** `Population` loads `_population.npz` (scale, w, λs, optional low-rank basis). Feedback for real users is `data/feedback/<user>.jsonl`; the rating app refits after every choice and logs `p_pred` *before* the choice, so its running accuracy is genuinely held-out. (Named `user_profile.py` because `profile.py` shadows a stdlib module.)

**Axes (`axes.py`).** Dynamic, not hard-coded: any phrase pair becomes an axis at runtime; image score = e · (mean(pos) − mean(neg)), z-scored over a sample. Custom axes persist in `data/axes.json`. The built-in template-paraphrase reliability score is optimistic; `verify.py paraphrase` (independently written wordings) is the real test.

**Conditioning (`conditional.py`).** "Given context C" = weight each pair by how strongly its images match C, refit population and user on weighted pairs, read out per-axis effects. The readout (`Conditioner.effect`) is **data-grounded**: slope of model score against axis position over real images. An earlier minimal-norm-direction readout under-reported effects ~10x; do not reintroduce it.

**Measured layers (`measured.py`).** Pixel stats (brightness, contrast, saturation, warmth, colourfulness, palette entropy, edge density, sharpness) and depth stats (subject size via Otsu split of depth, subject position, depth range, vertical gradient). The demo renders these layers per image.

## Findings that should steer future work

- Held-out accuracy (last 30% of each user's pairs by time): personal+prior 70.0%, population 68.5%, prompt-alignment only 65.0%, LAION aesthetic 47.2% (below chance), random 50%. Personal gain is +1.5 pts, not yet significant; grows with more training data (+0.6 at a 50% split → +2.4 at 80%). Calibration is good (ECE 0.029).
- Tuned priors land at the strongest value on offer (full-rank λ=1000, low-rank λ=100): Pick-a-Pic users' early choices only weakly predict later ones. Low-rank (k=32) was worse (67.4%). Active pair selection gave no gain over random in simulation.
- Soft conditioning as built *hurts* (log-loss +0.012, CI excludes 0) because each context fit discards data. The fair test (joint model: overall taste + context adjustment on all pairs) has not been run. Option D stays parked until it passes.
- Ranking: top-1 among ≥3 images for a prompt is 38–40% vs 21% random (154 groups).
- CLIP axes: reliable on style/subject (photo vs drawn, cute, fantasy, close-up). Unreliable under rewording on camera angle (0.22) and lighting (0.29). CLIP "dark vs bright" correlates only 0.30 with real pixel brightness: use measured features for brightness/saturation/contrast.
- Qwen2.5-VL-3B labels are near-constant for angle (97% eye-level), brightness (98% medium), saturation (97% vivid) and clutter (92% minimal); never use them as ground truth for those fields.
- Attribute sources (`reports/axes_check.csv`, AUC vs 2,000 Qwen labels): a logistic probe on CLIP trained with Qwen labels is best on every testable attribute (0.86–1.00; zero-shot 0.70–0.95, e.g. people 0.76 → 0.98, dramatic light 0.70 → 0.87). Depth alone gets close-up vs wide to 0.89 with no training, but measured layers add ≤ 0.005 on top of the probe. So: explanations should use the probe; measured layers are for literal brightness/saturation/contrast and for visualisation. All of this is agreement with a 3B VLM, not with humans; angle, darkness and saturation remain unverified.
- Depth extraction for 27k images took ~4 h wall-clock while sharing the GPU with the apps; budget accordingly.

## Gotchas

- OpenCV 5 removed Haar cascades (`cv2.CascadeClassifier`); face features were dropped for that reason.
- `hf_hub_download` stalled on this network; `build_dataset.py` uses `curl` (~13 MB/s measured).
- Pick-a-Pic contains junk prompts ("Gdhjyt"); app pools filter to ≥3 real words.
- `accuracy()` counts p == 0.5 as half-correct; otherwise constant baselines look like 35%.
