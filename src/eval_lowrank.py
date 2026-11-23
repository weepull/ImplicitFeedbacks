"""Low-rank personalisation vs. full-rank vs. population, same held-out protocol as evaluate.py.

  python src/eval_lowrank.py     -> prints tables, saves basis + chosen (k, lam) into _population.npz
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from taste import FEATS, Store, TasteProfile, accuracy, fit_lowrank, log_loss, make_basis, sigmoid  # noqa: E402
from user_profile import POP, Population  # noqa: E402

KS = (4, 8, 16, 32, 64)
LAMS = (0.3, 1, 3, 10, 30, 100)
CURVE = (0, 10, 25, 50, 100, 200, 300)


def main():
    store, pop = Store(), Population()
    pairs = pd.read_parquet("data/pairs_ready.parquet")
    Xg, _ = store.pair_diffs(pairs[pairs.role == "global"])
    Xg_s = Xg / pop.scale
    users = [(str(u), d.sort_values("created_at")) for u, d in pairs[pairs.role == "target"].groupby("user_id")]
    data = []
    for u, d in users:
        cut = int(0.7 * len(d))
        Xtr, ytr = store.pair_diffs(d.iloc[:cut]); Xte, yte = store.pair_diffs(d.iloc[cut:])
        data.append((u, Xtr / pop.scale, ytr, Xte / pop.scale, yte))

    # ---- tune (k, lam) on the last 20% of each user's TRAIN data only
    best = None
    bases = {k: make_basis(Xg_s, k) for k in KS}
    print("tuning (validation log-loss, train data only):")
    for k in KS:
        for lam in LAMS:
            ys, ps = [], []
            for u, Xtr, ytr, _, _ in data:
                c = int(0.8 * len(ytr))
                prof = fit_lowrank(u, Xtr[:c], ytr[:c], pop.w, bases[k], lam)
                ys.append(ytr[c:]); ps.append(prof.p_prefer(Xtr[c:]))
            ll = log_loss(np.concatenate(ys), np.concatenate(ps))
            if best is None or ll < best[0]:
                best = (ll, k, lam)
        print(f"  k={k:3d}  best-so-far k={best[1]} lam={best[2]} ll={best[0]:.4f}")
    _, K, LAM = best
    P = bases[K]
    print(f"chosen: k={K}, lam={LAM}  ->  {K + 2} personal numbers per user (vs {FEATS} full-rank)")

    # ---- held-out comparison
    rows, Y, PP, PF, PL = [], [], [], [], []
    for u, Xtr, ytr, Xte, yte in data:
        p_pop = sigmoid(Xte @ pop.w)
        p_full = TasteProfile.fit(u, Xtr, ytr, pop.w, pop.lam_prior).p_prefer(Xte)
        p_lr = fit_lowrank(u, Xtr, ytr, pop.w, P, LAM).p_prefer(Xte)
        rows.append(dict(user=u, n_train=len(ytr), population=accuracy(yte, p_pop),
                         full_rank=accuracy(yte, p_full), low_rank=accuracy(yte, p_lr)))
        Y.append(yte); PP.append(p_pop); PF.append(p_full); PL.append(p_lr)
    y, pp, pf, pl = map(np.concatenate, (Y, PP, PF, PL))
    print("\n== held-out (last 30% by time)")
    for name, p in (("population", pp), ("full-rank personal", pf), ("LOW-RANK personal", pl)):
        print(f"  {name:20s} acc {accuracy(y, p):.3f}  log-loss {log_loss(y, p):.4f}")
    rng = np.random.default_rng(0)
    for name, other in (("population", pp), ("full-rank", pf)):
        g = [accuracy(y[i], pl[i]) - accuracy(y[i], other[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(2000))]
        lo, hi = np.percentile(g, [2.5, 97.5])
        print(f"  low-rank minus {name:10s}: {np.mean(g):+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]")
    print("\nper user:")
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    # ---- learning curve
    print("\n== learning curve (mean test acc across users)")
    for n in CURVE:
        a_lr, a_f = [], []
        for u, Xtr, ytr, Xte, yte in data:
            if n > len(ytr):
                continue
            a_lr.append(accuracy(yte, fit_lowrank(u, Xtr[:n], ytr[:n], pop.w, P, LAM).p_prefer(Xte)))
            a_f.append(accuracy(yte, TasteProfile.fit(u, Xtr[:n], ytr[:n], pop.w, pop.lam_prior).p_prefer(Xte)))
        print(f"  n={n:3d}  low-rank {np.mean(a_lr):.3f}   full-rank {np.mean(a_f):.3f}   (users: {len(a_lr)})")

    # ---- planted-preference recovery (same test that showed ~15x shrinkage for full-rank)
    from conditional import Conditioner
    c = Conditioner(store, pop)
    u, Xtr, ytr, Xte, yte = data[0]
    Xs = np.vstack([Xtr, Xte])
    print("\n== planted preference recovery (planted +1.0 log-odds per std; 5 resamples)")
    for ax in ("close-up vs wide", "dark vs bright", "photoreal vs illustrated"):
        a = c.reg.get(ax); dt = a["direction"]
        d = np.zeros(FEATS); d[:768] = dt / pop.scale[:768] * a["std"] / (dt @ dt)
        w_true = pop.w + 1.0 * d / (d @ d)
        base = c.effect(TasteProfile("p", pop.w, np.eye(FEATS), 0), ax)[0]
        r_lr, r_f = [], []
        for s in range(5):
            yy = (np.random.default_rng(s).random(len(Xs)) < sigmoid(Xs @ w_true)).astype(float)
            r_lr.append(c.effect(fit_lowrank("s", Xs, yy, pop.w, P, LAM), ax)[0] - base)
            r_f.append(c.effect(TasteProfile.fit("s", Xs, yy, pop.w, pop.lam_prior), ax)[0] - base)
        print(f"  {ax:26s} low-rank recovers {np.mean(r_lr):+.3f}   full-rank recovers {np.mean(r_f):+.3f}")

    # ---- save for the app / reranker
    z = dict(np.load(POP))
    z.update(basis=P, lr_k=K, lr_lam=LAM)
    np.savez(POP, **z)
    print(f"\nsaved basis (k={K}, lam={LAM}) into {POP}")


if __name__ == "__main__":
    main()
