"""Rerank candidate images for a user.

  python src/rerank.py --user abc --prompt "a knight fighting a dragon" path/to/*.jpg
  python src/rerank.py --user abc --dir some/folder

Ranking score = the user's full taste score (taste + prompt adherence + aesthetic weights).
P(like)       = P(user would pick this image over a random generated image), taste only,
                with an 80% credible interval. A user with no ratings gets the population
                prior, so the interval is wide; it narrows as they rate.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from clip_model import Aesthetic, Clip  # noqa: E402
from conditional import Conditioner, user_pairs  # noqa: E402
from taste import EMB_DIM, Store, TasteProfile  # noqa: E402
from user_profile import Population  # noqa: E402


def reference_pool(store: Store, pop: Population, n=1000, seed=0):
    """Taste-only features (alignment zeroed) for a random sample of dataset images."""
    rng = np.random.default_rng(seed)
    ii = rng.choice(len(store.uids), min(n, len(store.uids)), replace=False)
    X = np.hstack([store.emb[ii], np.zeros((len(ii), 1)), store.aest[ii, None]])
    return X / pop.scale


def user_profile_for(user, given, cond):
    """Plain or context-conditioned profile; falls back to the (conditioned) population."""
    pairs = user_pairs(user)
    if len(pairs) == 0:
        prof = cond.population_in(given) if given else cond.pop.cold_profile(user)
        return TasteProfile(user, prof.mean, np.eye(len(prof.mean)) / cond.pop.lam_prior, 0)
    prof, _ = cond.user_in(user, pairs, given)
    return prof


def reasons(prof, cond, emb, top=3, min_contrib=0.05):
    """Per image: which clearly-held preferences this image hits or misses.
    contribution = (user's effect per std on axis) x (image's std position on axis)."""
    axes = [n for n in cond.reg.names() if not n.startswith("ctx:")]
    eff = {a: cond.effect(prof, a) for a in axes}
    eff = {a: e for a, e in eff.items() if (e[1] > 0 or e[2] < 0)}  # only clear effects
    out = []
    for i in range(len(emb)):
        contrib = []
        for a, (m, *_rest) in eff.items():
            z = float(cond.reg.z(a, emb[i:i + 1])[0])
            pos, neg = a.split(" vs ") if " vs " in a else (a, "not " + a)
            contrib.append((m * z, pos if z > 0 else neg))
        contrib = [c for c in sorted(contrib, key=lambda c: -abs(c[0])) if abs(c[0]) >= min_contrib][:top]
        out.append("  ".join(f"{'+' if c > 0 else '−'} {label}" for c, label in contrib) or "(no clear drivers)")
    return out


def rerank(user, files, prompt=None, given=None, clip=None, store=None, pop=None):
    clip, store, pop = clip or Clip(), store or Store(), pop or Population()
    cond = Conditioner(store, pop, clip)
    prof = user_profile_for(user, given, cond)
    emb = clip.embed_images(files)
    aest = Aesthetic().score(emb)
    align = (emb @ clip.embed_texts([prompt])[0]) if prompt else np.zeros(len(files))
    X = np.hstack([emb, align[:, None], aest[:, None]]) / pop.scale
    X_taste = X.copy()
    X_taste[:, EMB_DIM] = 0
    p, lo, hi = prof.p_like(X_taste, reference_pool(store, pop))
    df = pd.DataFrame({"file": [os.path.basename(f) for f in files], "score": prof.score(X),
                       "p_like": p, "p_lo": lo, "p_hi": hi,
                       "prompt_align": align, "aesthetic": aest, "why": reasons(prof, cond, emb)})
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    df.index += 1
    return df, prof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True)
    ap.add_argument("--prompt")
    ap.add_argument("--given", help="context to condition taste on, e.g. 'photorealistic'")
    ap.add_argument("--dir")
    ap.add_argument("files", nargs="*")
    a = ap.parse_args()
    files = a.files or sorted(glob.glob(os.path.join(a.dir, "*.jp*g")) + glob.glob(os.path.join(a.dir, "*.png")))
    df, prof = rerank(a.user, files, a.prompt, a.given)
    print(f"user={a.user}  profile built from {prof.n_pairs} comparisons"
          + (f"  context: {a.given!r}" if a.given else "")
          + ("  (cold start: population prior only)" if prof.n_pairs == 0 else ""))
    for i, r in df.iterrows():
        print(f"{i:3d}. {r.file:40s} P(like)={r.p_like:.2f} [{r.p_lo:.2f}-{r.p_hi:.2f}]  "
              f"align={r.prompt_align:.3f}  aesthetic={r.aesthetic:.2f}")
        print(f"     {r.why}")


if __name__ == "__main__":
    main()
