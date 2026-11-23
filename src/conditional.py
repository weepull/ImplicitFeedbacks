"""Soft-conditioned taste: "given context C, what does user U prefer?"

1. Context C is any phrase ("photorealistic", "a night scene") or contrast ("A vs B").
   Every image gets a continuous score on C (see axes.py); a pair's weight is how
   strongly its images match C (0..1), so half-matching pairs count as half.
2. Prior = the population refit with the same weights (how the *average* user's taste
   shifts in C). The user's model is refit on their weighted pairs on top of that prior,
   so no data is counted twice and thin contexts fall back to "average user in C".
3. Readout: how much the model's score changes as REAL images move +1 std along each axis
   (log-odds), with a 90% interval from the Laplace posterior and the effective number of
   pairs behind it. Correlated axes (dark / moody / low-saturation) share credit.

  python src/conditional.py --user 3064 --given "photorealistic"
  python src/conditional.py --user abc --given "portrait vs landscape" --axes "close-up vs wide"
  python src/conditional.py --add "cinematic vs snapshot=a cinematic film still=a casual phone snapshot"
      (either side may list paraphrases separated by |)
  python src/conditional.py --eval        # go/no-go: does conditioning predict better?
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from axes import AxisRegistry  # noqa: E402
from taste import EMB_DIM, FEATS, Store, TasteProfile, accuracy, fit_map, log_loss, sigmoid  # noqa: E402
from user_profile import Population, load_feedback  # noqa: E402

COVERAGE = 0.3     # by default C covers roughly the top 30% of images
SHARPNESS = 3.0    # how quickly weight goes 0 -> 1 around the threshold (in std units)
GENERIC = "an image | a picture | a photo"


class Conditioner:
    def __init__(self, store: Store, pop: Population, clip=None):
        if clip is None:
            from clip_model import Clip
            clip = Clip()
        self.store, self.pop = store, pop
        self.reg = AxisRegistry(clip, store.emb)
        pairs = pd.read_parquet("data/pairs_ready.parquet")
        self.glob = pairs[pairs.role == "global"]
        self.Xg, self.yg = store.pair_diffs(self.glob)
        self._pop_cache: dict = {}
        # real-image sample for the data-grounded readout (taste features, no prompt context)
        ii = np.random.default_rng(0).choice(len(store.uids), min(4000, len(store.uids)), replace=False)
        self._img_emb = store.emb[ii]
        self._img_X = np.hstack([store.emb[ii], np.zeros((len(ii), 1)), store.aest[ii, None]]) / pop.scale
        self._g: dict = {}

    # ---- context -> weights
    def context_axis(self, cond: str) -> str:
        if cond in self.reg.axes:
            return cond
        if " vs " in cond:
            pos, neg = cond.split(" vs ", 1)
        else:
            pos, neg = cond, GENERIC
        name = f"ctx:{cond}"
        if name not in self.reg.axes:
            self.reg.add(name, pos, neg, persist=False)
        return name

    def image_weight(self, cond: str, emb: np.ndarray, coverage=COVERAGE) -> np.ndarray:
        name = self.context_axis(cond)
        z_sample = self.reg.z(name, self.reg.sample)
        thr = np.quantile(z_sample, 1 - coverage)
        return sigmoid(SHARPNESS * (self.reg.z(name, emb) - thr))

    def pair_weight(self, cond: str, pairs: pd.DataFrame, coverage=COVERAGE) -> np.ndarray:
        ea = self.store.emb[[self.store.idx[u] for u in pairs.image_0_uid]]
        eb = self.store.emb[[self.store.idx[u] for u in pairs.image_1_uid]]
        return 0.5 * (self.image_weight(cond, ea, coverage) + self.image_weight(cond, eb, coverage))

    # ---- fits
    def population_in(self, cond: str):
        if cond not in self._pop_cache:
            sw = self.pair_weight(cond, self.glob)
            w, cov = fit_map(self.Xg / self.pop.scale, self.yg, np.zeros(FEATS), self.pop.lam, sample_weight=sw)
            self._pop_cache[cond] = TasteProfile("population", w, cov, len(self.yg))
        return self._pop_cache[cond]

    def user_in(self, user: str, pairs: pd.DataFrame, cond: str | None):
        X, y = self.store.pair_diffs(pairs)
        Xs = X / self.pop.scale
        if cond is None:
            return TasteProfile.fit(user, Xs, y, self.pop.w, self.pop.lam_prior), float(len(y))
        sw = self.pair_weight(cond, pairs)
        prior = self.population_in(cond).mean
        prof = TasteProfile.fit(user, Xs, y, prior, self.pop.lam_prior, sample_weight=sw)
        n_eff = float(sw.sum() ** 2 / (sw ** 2).sum()) if sw.sum() > 0 else 0.0
        return prof, n_eff

    # ---- readout
    def effect(self, prof: TasteProfile, axis: str):
        """Data-grounded readout: the slope of the model's score against the axis position over
        REAL images (log-odds change per +1 std along the axis), with a 90% interval.

        Linear in w: effect = g . w with g = X_img^T (z - mean z) / ((n-1) var z). An earlier
        version measured w along an idealised CLIP direction that real images barely vary
        along, which made every effect look ~10x smaller than it is (see reports/verify.txt).
        Correlated axes share credit, because real images move along them together."""
        if axis not in self._g:
            z = self.reg.z(axis, self._img_emb)
            zc = z - z.mean()
            self._g[axis] = self._img_X.T @ zc / ((len(z) - 1) * z.var(ddof=1))
        g = self._g[axis]
        mu = float(g @ prof.mean)
        sd = float(np.sqrt(max(g @ prof.cov @ g, 0.0)))
        return mu, mu - 1.645 * sd, mu + 1.645 * sd, self.reg.get(axis)["reliable"]

    def readout(self, user, pairs, cond, axes=None):
        axes = axes or [n for n in self.reg.names() if not n.startswith("ctx:")]
        pc, n_eff = self.user_in(user, pairs, cond)
        pu, _ = self.user_in(user, pairs, None)
        pp = self.population_in(cond)
        rows = []
        for ax in axes:
            m, lo, hi, rel = self.effect(pc, ax)
            rows.append(dict(axis=ax, effect=m, lo=lo, hi=hi,
                             user_overall=self.effect(pu, ax)[0],
                             avg_user_in_context=self.effect(pp, ax)[0],
                             clear=bool(lo > 0 or hi < 0), reliable_axis=rel))
        return pd.DataFrame(rows).sort_values("effect", key=abs, ascending=False), n_eff


def user_pairs(user: str) -> pd.DataFrame:
    pairs = pd.read_parquet("data/pairs_ready.parquet")
    if user in set(pairs.user_id.astype(str)):
        return pairs[pairs.user_id.astype(str) == user]
    fb = load_feedback(user)
    fb = fb[fb.chosen.notna()] if len(fb) else fb
    return pd.DataFrame({"caption": fb.caption, "image_0_uid": fb.left, "image_1_uid": fb.right,
                         "label_0": (fb.chosen == fb.left).astype(float), "created_at": fb.ts})


def print_readout(user, cond, df, n_eff):
    print(f"\n{user} | given: {cond!r} | effective pairs in context: {n_eff:.0f}")
    print("effect = change in log-odds of preferring an image that is 1 std further toward the axis's first phrase")
    print(f"{'axis':28s} {'in context':>24s} {'user overall':>13s} {'avg user here':>14s}")
    for _, r in df.iterrows():
        tag = "CLEAR" if r.clear else "  -  "
        warn = "" if r.reliable_axis else "  (axis wording-sensitive)"
        print(f"{r.axis:28s} {r.effect:+.2f} [{r.lo:+.2f},{r.hi:+.2f}] {tag} {r.user_overall:+13.2f} "
              f"{r.avg_user_in_context:+14.2f}{warn}")


def evaluate(cond_list=None):
    """Go/no-go for the neural conditional model: does conditioning improve held-out prediction?"""
    store, pop = Store(), Population()
    c = Conditioner(store, pop)
    cond_list = cond_list or ["a photorealistic photograph", "anime style artwork", "a portrait of a person",
                              "a landscape scene", "a dark, moody image", "an epic fantasy world",
                              "a cute cartoon character", "a 3D render"]
    pairs = pd.read_parquet("data/pairs_ready.parquet")
    rows = []
    for user, df in pairs[pairs.role == "target"].groupby("user_id"):
        df = df.sort_values("created_at")
        cut = int(0.7 * len(df))
        tr, te = df.iloc[:cut], df.iloc[cut:]
        Xte, yte = store.pair_diffs(te)
        Xte = Xte / pop.scale
        base, _ = c.user_in(str(user), tr, None)
        p_base = base.p_prefer(Xte)
        for cond in cond_list:
            prof, n_eff = c.user_in(str(user), tr, cond)
            sw = c.pair_weight(cond, te)
            p_c = prof.p_prefer(Xte)
            rows.append(dict(user=user, cond=cond, n_eff_train=n_eff, w_test=sw.sum(),
                             y=yte, sw=sw, p_base=p_base, p_cond=p_c))
    print("\n== Does conditioning help? Held-out pairs, weighted by how well they match the context")
    print(f"{'context':30s} {'eff. test pairs':>15s} {'acc plain':>10s} {'acc cond':>9s} {'logloss plain':>14s} {'logloss cond':>13s}")
    out = []
    for cond in cond_list:
        rs = [r for r in rows if r["cond"] == cond]
        y = np.concatenate([r["y"] for r in rs]); sw = np.concatenate([r["sw"] for r in rs])
        pb = np.concatenate([r["p_base"] for r in rs]); pc = np.concatenate([r["p_cond"] for r in rs])
        wa = lambda p: float((sw * ((p > 0.5) == (y > 0.5))).sum() / sw.sum())  # noqa: E731
        wl = lambda p: float((sw * -(y * np.log(np.clip(p, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-6, 1)))).sum() / sw.sum())  # noqa: E731
        out.append(dict(cond=cond, n=sw.sum(), acc_plain=wa(pb), acc_cond=wa(pc), ll_plain=wl(pb), ll_cond=wl(pc)))
        o = out[-1]
        print(f"{cond:30s} {o['n']:15.0f} {o['acc_plain']:10.3f} {o['acc_cond']:9.3f} {o['ll_plain']:14.4f} {o['ll_cond']:13.4f}")
    # pooled paired bootstrap on log-loss difference (negative = conditioning helps)
    y = np.concatenate([r["y"] for r in rows]); sw = np.concatenate([r["sw"] for r in rows])
    lb = -(y * np.log(np.concatenate([r["p_base"] for r in rows]).clip(1e-6)) + (1 - y) * np.log((1 - np.concatenate([r["p_base"] for r in rows])).clip(1e-6)))
    lc = -(y * np.log(np.concatenate([r["p_cond"] for r in rows]).clip(1e-6)) + (1 - y) * np.log((1 - np.concatenate([r["p_cond"] for r in rows])).clip(1e-6)))
    rng = np.random.default_rng(0)
    diffs = []
    for _ in range(2000):
        i = rng.integers(0, len(y), len(y))
        diffs.append(((lc[i] - lb[i]) * sw[i]).sum() / sw[i].sum())
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    print(f"\npooled weighted log-loss change (cond - plain): {np.mean(diffs):+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
    print("verdict:", "conditioning HELPS -> green light for option D (docs/roadmap.md)" if hi < 0 else
          "conditioning HURTS" if lo > 0 else "no measurable difference yet -> context effects not established")
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user")
    ap.add_argument("--given", help="context phrase, or 'A vs B'")
    ap.add_argument("--axes", nargs="*", help="axis names to read out (default: all registered)")
    ap.add_argument("--add", help="register an axis: 'name=pos phrase|...=neg phrase|...' or 'name=pos=neg'")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--eval", action="store_true")
    a = ap.parse_args()
    if a.eval:
        evaluate()
        return
    store, pop = Store(), Population()
    c = Conditioner(store, pop)
    if a.add:
        name, pos, neg = a.add.split("=", 2)
        ax = c.reg.add(name.strip(), pos.strip(), neg.strip())
        print(f"added axis {name!r}: reliability {ax['reliability']:.2f} "
              f"({'ok' if ax['reliable'] else 'wording-sensitive, rephrase or add paraphrases with |'})")
    if a.list:
        for n in c.reg.names():
            ax = c.reg.get(n)
            print(f"{n:28s} reliability {ax['reliability']:.2f} {'' if ax['reliable'] else '(wording-sensitive)'}")
    if a.user and a.given:
        df, n_eff = c.readout(a.user, user_pairs(a.user), a.given, a.axes)
        print_readout(a.user, a.given, df, n_eff)


if __name__ == "__main__":
    main()
