"""Evaluate taste models on stand-in users (Pick-a-Pic heavy raters).

Per target user: time-ordered split, first 70% train / last 30% test.
Methods compared on the test pairs (pairwise accuracy, log loss, calibration):
  random, prompt-alignment (zero-training), LAION aesthetic (zero-training),
  population (other users only), personal-only (no prior), personal + population prior.
Also a learning curve: accuracy vs. number of the user's own training pairs.

Usage: python src/evaluate.py            -> prints tables, writes reports/eval.json
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from taste import FEATS, Population, Store, TasteProfile, accuracy, ece, fit_map, log_loss, sigmoid  # noqa: E402

LAMS = (1, 3, 10, 30, 100, 300, 1000)
CURVE = (0, 10, 25, 50, 100, 200, 400)


def split_user(df):
    df = df.sort_values("created_at")
    cut = int(0.7 * len(df))
    return df.iloc[:cut], df.iloc[cut:]


def tune_personal_lam(store, pop, targets, prior: bool):
    """Pick one lam for all users using only train data (last 20% of train as validation)."""
    best = None
    for lam in LAMS:
        ys, ps = [], []
        for _, df in targets:
            tr, _ = split_user(df)
            cut = int(0.8 * len(tr))
            X, y = store.pair_diffs(tr)
            Xs = X / pop.scale
            m = pop.w if prior else np.zeros(FEATS)
            prof = TasteProfile.fit("", Xs[:cut], y[:cut], m, lam)
            ys.append(y[cut:]); ps.append(prof.p_prefer(Xs[cut:]))
        ll = log_loss(np.concatenate(ys), np.concatenate(ps))
        if best is None or ll < best[0]:
            best = (ll, lam)
    return best[1]


def main():
    store = Store()
    pairs = pd.read_parquet("data/pairs_ready.parquet")
    glob_pairs = pairs[pairs.role == "global"]
    targets = [(u, d) for u, d in pairs[pairs.role == "target"].groupby("user_id")]
    print(f"global pairs: {len(glob_pairs)}  target users: {len(targets)}  "
          f"pairs/user: {[len(d) for _, d in targets]}")

    Xg, yg = store.pair_diffs(glob_pairs)
    pop = Population.fit(Xg, yg)
    print(f"population lam={pop.lam}")
    lam_prior = tune_personal_lam(store, pop, targets, prior=True)
    lam_solo = tune_personal_lam(store, pop, targets, prior=False)
    print(f"personal lam (with prior)={lam_prior}  (no prior)={lam_solo}")

    rows, curve = [], []
    all_y, all_p = [], {}
    for user, df in targets:
        tr, te = split_user(df)
        Xtr, ytr = store.pair_diffs(tr)
        Xte, yte = store.pair_diffs(te)
        Xtr_s, Xte_s = Xtr / pop.scale, Xte / pop.scale
        preds = {
            "random": np.full(len(yte), 0.5),
            "prompt_alignment": sigmoid(50 * Xte[:, FEATS - 2]),
            "laion_aesthetic": sigmoid(Xte[:, FEATS - 1]),
            "population": sigmoid(Xte_s @ pop.w),
            "personal_only": TasteProfile.fit(user, Xtr_s, ytr, np.zeros(FEATS), lam_solo).p_prefer(Xte_s),
            "personal+prior": TasteProfile.fit(user, Xtr_s, ytr, pop.w, lam_prior).p_prefer(Xte_s),
        }
        for m, p in preds.items():
            rows.append(dict(user=user, method=m, acc=accuracy(yte, p), n_test=len(yte), n_train=len(ytr)))
            all_p.setdefault(m, []).append(p)
        all_y.append(yte)
        for n in CURVE:
            if n > len(ytr):
                continue
            for m, prior, lam in (("personal+prior", pop.w, lam_prior), ("personal_only", np.zeros(FEATS), lam_solo)):
                p = TasteProfile.fit(user, Xtr_s[:n], ytr[:n], prior, lam).p_prefer(Xte_s)
                curve.append(dict(user=user, method=m, n=n, acc=accuracy(yte, p)))

    res = pd.DataFrame(rows)
    y = np.concatenate(all_y)
    summary = []
    for m, ps in all_p.items():
        p = np.concatenate(ps)
        per_user = res[res.method == m].acc
        summary.append(dict(method=m, pooled_acc=accuracy(y, p), mean_user_acc=per_user.mean(),
                            min_user=per_user.min(), max_user=per_user.max(),
                            log_loss=log_loss(y, p), ece=ece(y, p) if m != "random" else np.nan))
    summary = pd.DataFrame(summary)
    print("\n== held-out pairwise accuracy (last 30% of each user's pairs, by time)")
    print(summary.round(3).to_string(index=False))

    print("\n== per user accuracy")
    print(res.pivot(index="user", columns="method", values="acc").round(3)
          .assign(n_train=res.groupby("user").n_train.first()).to_string())

    cv = pd.DataFrame(curve).groupby(["method", "n"]).acc.mean().unstack(0)
    print("\n== learning curve: mean test accuracy vs. # of user's own training pairs")
    print(cv.round(3).to_string())

    # bootstrap CI on the personal-vs-population gap (paired, resample test pairs)
    pp, po = np.concatenate(all_p["personal+prior"]), np.concatenate(all_p["population"])
    rng = np.random.default_rng(0)
    gaps = []
    for _ in range(2000):
        i = rng.integers(0, len(y), len(y))
        gaps.append(accuracy(y[i], pp[i]) - accuracy(y[i], po[i]))
    lo, hi = np.percentile(gaps, [2.5, 97.5])
    print(f"\npersonal+prior minus population: {np.mean(gaps):+.3f} (95% CI {lo:+.3f} .. {hi:+.3f})")

    os.makedirs("reports", exist_ok=True)
    json.dump(dict(summary=summary.to_dict("records"), per_user=res.to_dict("records"),
                   curve=pd.DataFrame(curve).to_dict("records"),
                   gap=dict(mean=float(np.mean(gaps)), lo=float(lo), hi=float(hi)),
                   lam=dict(population=pop.lam, prior=lam_prior, solo=lam_solo)),
              open("reports/eval.json", "w"), indent=1, default=float)

    # persist the population model for the rating app / reranker
    os.makedirs("data/profiles", exist_ok=True)
    np.savez("data/profiles/_population.npz", scale=pop.scale, w=pop.w, lam=pop.lam, lam_prior=lam_prior)


if __name__ == "__main__":
    main()
