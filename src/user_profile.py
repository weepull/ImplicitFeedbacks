"""Per-user taste profiles: fit from collected feedback, describe in words, persist.

Feedback lives in data/feedback/<user>.jsonl, one line per comparison:
  {"caption", "left", "right", "chosen", "response_ms", "p_pred", "ts"}
("chosen" is a uid, or null for skip). Profiles in data/profiles/<user>.npz.

Usage:
  python src/user_profile.py fit abc         # refit from feedback, print description
  python src/user_profile.py describe abc
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from taste import EMB_DIM, FEATS, Store, TasteProfile  # noqa: E402

POP = "data/profiles/_population.npz"

# Axes to read a taste vector against live in axes.py (dynamic registry).
from axes import STARTER as AXES  # noqa: E402


class Population:
    def __init__(self, path=POP):
        z = np.load(path)
        self.scale, self.w = z["scale"], z["w"]
        self.lam, self.lam_prior = float(z["lam"]), float(z["lam_prior"])

    def cold_profile(self, user):
        return TasteProfile(user, self.w.copy(), np.eye(FEATS) / self.lam_prior, 0)


def feedback_path(user):
    return f"data/feedback/{user}.jsonl"


def profile_path(user):
    return f"data/profiles/{user}.npz"


def load_feedback(user) -> pd.DataFrame:
    p = feedback_path(user)
    if not os.path.exists(p):
        return pd.DataFrame(columns=["caption", "left", "right", "chosen", "response_ms", "p_pred", "ts"])
    return pd.DataFrame([json.loads(l) for l in open(p) if l.strip()])


def fit_from_feedback(user, store: Store, pop: Population) -> TasteProfile:
    fb = load_feedback(user)
    fb = fb[fb.chosen.notna()] if len(fb) else fb
    if len(fb) == 0:
        return pop.cold_profile(user)
    pairs = pd.DataFrame({
        "caption": fb.caption, "image_0_uid": fb.left, "image_1_uid": fb.right,
        "label_0": (fb.chosen == fb.left).astype(float),
    })
    X, y = store.pair_diffs(pairs)
    return TasteProfile.fit(user, X / pop.scale, y, pop.w, pop.lam_prior)


def load_profile(user, store=None, pop=None) -> TasteProfile:
    if os.path.exists(profile_path(user)):
        return TasteProfile.load(profile_path(user))
    return (pop or Population()).cold_profile(user)


def describe(prof: TasteProfile, pop: Population, clip=None, relative=True):
    """Project the taste vector onto text axes. relative=True: vs. population average."""
    if clip is None:
        from clip_model import Clip
        clip = Clip()
    w = prof.mean - (pop.w if relative else 0)
    w_img = w[:EMB_DIM] / pop.scale[:EMB_DIM]  # back to raw embedding space
    names = list(AXES)
    pos = clip.embed_texts([AXES[n][0] for n in names])
    neg = clip.embed_texts([AXES[n][1] for n in names])
    s = (pos - neg) @ w_img
    s = s / (np.abs(s).max() + 1e-12)
    out = pd.DataFrame({"axis": names, "lean": s}).sort_values("lean", ascending=False)
    extras = {
        "weight on prompt adherence": w[EMB_DIM] , "weight on LAION aesthetic": w[EMB_DIM + 1],
    }
    return out, extras


def main():
    cmd, user = sys.argv[1], sys.argv[2]
    store, pop = Store(), Population()
    if cmd == "fit":
        prof = fit_from_feedback(user, store, pop)
        os.makedirs("data/profiles", exist_ok=True)
        prof.save(profile_path(user))
        print(f"fitted {user} on {prof.n_pairs} comparisons -> {profile_path(user)}")
    else:
        prof = load_profile(user, pop=pop)
    d, extras = describe(prof, pop)
    print(f"\n{user}'s taste vs. the average Pick-a-Pic user (+1 = leans to the first phrase):")
    for _, r in d.iterrows():
        bar = "#" * int(abs(r.lean) * 20)
        print(f"  {r.axis:26s} {r.lean:+.2f} {'+' if r.lean > 0 else '-'}{bar}")
    for k, v in extras.items():
        print(f"  {k}: {v:+.3f} (relative to population)")


if __name__ == "__main__":
    main()
