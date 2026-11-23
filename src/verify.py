"""Verification checks for things that were only smoke-tested.

  python src/verify.py splits      # A: is the personal-vs-population result robust to the split point?
  python src/verify.py rank        # B: reranker as a ranker (top-1 among a user's images for one prompt)
  python src/verify.py readout     # C: does the axis readout recover planted preferences / agree with data?
  python src/verify.py paraphrase  # D: do genuinely different wordings of an axis agree?
  python src/verify.py app         # E: rating app over HTTP + active vs random pair selection
  python src/verify.py all
Results are appended to reports/verify.txt.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from axes import STARTER  # noqa: E402
from conditional import Conditioner  # noqa: E402
from taste import FEATS, TasteProfile, Store, accuracy, sigmoid  # noqa: E402
from user_profile import Population  # noqa: E402

OUT = "reports/verify.txt"


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    os.makedirs("reports", exist_ok=True)
    with open(OUT, "a") as f:
        f.write(s + "\n")


def load():
    store, pop = Store(), Population()
    pairs = pd.read_parquet("data/pairs_ready.parquet")
    return store, pop, pairs


def targets(pairs):
    return [(str(u), d.sort_values("created_at")) for u, d in pairs[pairs.role == "target"].groupby("user_id")]


# ------------------------------------------------------------------ A
def check_splits():
    store, pop, pairs = load()
    log("\n== A. Split robustness: personal+prior vs population at different time cut points")
    for frac in (0.5, 0.6, 0.7, 0.8):
        y_all, pp, po = [], [], []
        for u, df in targets(pairs):
            cut = int(frac * len(df))
            Xtr, ytr = store.pair_diffs(df.iloc[:cut]); Xte, yte = store.pair_diffs(df.iloc[cut:])
            prof = TasteProfile.fit(u, Xtr / pop.scale, ytr, pop.w, pop.lam_prior)
            y_all.append(yte); pp.append(prof.p_prefer(Xte / pop.scale)); po.append(sigmoid(Xte / pop.scale @ pop.w))
        y, pp, po = map(np.concatenate, (y_all, pp, po))
        rng = np.random.default_rng(0)
        gaps = [accuracy(y[i], pp[i]) - accuracy(y[i], po[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(1000))]
        lo, hi = np.percentile(gaps, [2.5, 97.5])
        log(f"  train first {frac:.0%}: n_test={len(y):5d}  population {accuracy(y, po):.3f}  personal+prior "
            f"{accuracy(y, pp):.3f}  gap {np.mean(gaps):+.3f} [{lo:+.3f},{hi:+.3f}]")


# ------------------------------------------------------------------ B
def check_rank():
    store, pop, pairs = load()
    log("\n== B. Reranker as ranker: prompts where a user compared >=3 images (test period only)")
    log("   ground truth = image with the most wins in that user's comparisons for that prompt (ties dropped)")
    res = {"random": [], "prompt_alignment": [], "population": [], "personal+prior": []}
    n_groups = 0
    for u, df in targets(pairs):
        cut = int(0.7 * len(df))
        tr, te = df.iloc[:cut], df.iloc[cut:]
        Xtr, ytr = store.pair_diffs(tr)
        prof = TasteProfile.fit(u, Xtr / pop.scale, ytr, pop.w, pop.lam_prior)
        for cap, g in te.groupby("caption"):
            wins = {}
            for _, r in g.iterrows():
                w, l = (r.image_0_uid, r.image_1_uid) if r.label_0 > 0.5 else (r.image_1_uid, r.image_0_uid)
                wins[w] = wins.get(w, 0) + 1
                wins.setdefault(l, 0)
            if len(wins) < 3:
                continue
            uids = list(wins)
            best = max(wins.values())
            if sum(v == best for v in wins.values()) > 1:
                continue
            truth = uids[int(np.argmax([wins[x] for x in uids]))]
            X = store.features(uids, [cap] * len(uids)) / pop.scale
            n_groups += 1
            res["random"].append(1 / len(uids))
            res["prompt_alignment"].append(uids[int(np.argmax(X[:, FEATS - 2]))] == truth)
            res["population"].append(uids[int(np.argmax(X @ pop.w))] == truth)
            res["personal+prior"].append(uids[int(np.argmax(X @ prof.mean))] == truth)
    log(f"   groups: {n_groups}")
    for k, v in res.items():
        v = np.array(v, float)
        se = v.std() / np.sqrt(len(v)) if len(v) else np.nan
        log(f"   top-1 hit rate  {k:18s} {v.mean():.3f} ± {se:.3f}")


# ------------------------------------------------------------------ C
def check_readout():
    store, pop, pairs = load()
    c = Conditioner(store, pop)
    axes = list(STARTER)
    log("\n== C1. Planted-preference recovery: synthetic user = population + known effect on one axis")
    log("   labels resampled on a real user's pairs; does the readout return the planted effect?")
    rng = np.random.default_rng(0)
    u, df = targets(pairs)[0]
    X, _ = store.pair_diffs(df)
    Xs = X / pop.scale
    for ax in ("close-up vs wide", "dark vs bright", "photoreal vs illustrated"):
        a = c.reg.get(ax)
        dt = a["direction"]
        d = np.zeros(FEATS); d[:768] = dt / pop.scale[:768] * a["std"] / (dt @ dt)
        for beta in (0.0, 0.5, 1.0):
            # add beta along d: choose w_true = pop.w + beta * d / (d.d) so that d.w_true = d.pop.w + beta
            w_true = pop.w + beta * d / (d @ d)
            rec = []
            for s in range(5):
                y = (rng.random(len(Xs)) < sigmoid(Xs @ w_true)).astype(float)
                prof = TasteProfile.fit("syn", Xs, y, pop.w, pop.lam_prior)
                rec.append(c.effect(prof, ax)[0] - c.effect(TasteProfile("p", pop.w, prof.cov, 0), ax)[0])
            others = []
            for ox in axes:
                if ox == ax:
                    continue
                others.append(abs(c.effect(prof, ox)[0] - c.effect(TasteProfile("p", pop.w, prof.cov, 0), ox)[0]))
            log(f"   {ax:26s} planted {beta:+.1f} -> recovered {np.mean(rec):+.3f} (sd {np.std(rec):.3f}); "
                f"mean |leak| into other axes {np.mean(others):.3f}  [n_pairs={len(y)}]")

    log("\n== C2. Real-data agreement: for each user x axis, does the image further toward the axis win more often")
    log("   in the user's held-out pairs, in the direction the readout says? (only pairs with |dz| > 0.5)")
    agree, tot, corr_m, corr_e = 0, 0, [], []
    for u, df in targets(pairs):
        cut = int(0.7 * len(df))
        tr, te = df.iloc[:cut], df.iloc[cut:]
        Xtr, ytr = store.pair_diffs(tr)
        prof = TasteProfile.fit(u, Xtr / pop.scale, ytr, pop.w, pop.lam_prior)
        ea = store.emb[[store.idx[x] for x in te.image_0_uid]]
        eb = store.emb[[store.idx[x] for x in te.image_1_uid]]
        yte = (te.label_0.values > 0.5).astype(float)
        for ax in axes:
            m, lo, hi, _ = c.effect(prof, ax)
            dz = c.reg.z(ax, ea) - c.reg.z(ax, eb)
            sel = np.abs(dz) > 0.5
            if sel.sum() < 20:
                continue
            emp = float(((dz[sel] > 0) == (yte[sel] > 0.5)).mean()) - 0.5  # >0: further-toward-pos image wins more
            corr_m.append(m); corr_e.append(emp)
            if lo > 0 or hi < 0:  # only judge clear readouts
                tot += 1
                agree += int(np.sign(m) == np.sign(emp))
    log(f"   clear readouts whose sign matches held-out behaviour: {agree}/{tot}")
    log(f"   correlation(readout effect, held-out win-rate excess) over all user x axis: "
        f"{np.corrcoef(corr_m, corr_e)[0, 1]:.2f}  (n={len(corr_m)})")


# ------------------------------------------------------------------ D
ALT = {  # genuinely different wordings written for each starter axis
    "photoreal vs illustrated": ("a real camera photo of the scene", "a hand-drawn or digitally painted artwork"),
    "anime vs western": ("Japanese manga / anime character art", "American cartoon or European oil painting"),
    "dark vs bright": ("shadowy, dim, night-like tones", "sunny, well-lit, luminous tones"),
    "saturated vs muted": ("intense, colourful, neon palette", "greyish, pastel, washed-out palette"),
    "dramatic vs flat light": ("strong shadows and spotlight, chiaroscuro", "soft overcast light without shadows"),
    "busy vs minimal": ("cluttered scene full of many objects", "sparse composition with lots of negative space"),
    "close-up vs wide": ("the subject fills the frame, seen up close", "a distant view showing the whole environment"),
    "low vs eye-level angle": ("camera on the ground pointing upward at the subject", "camera at head height, straight on"),
    "action vs calm": ("fast movement, fighting or running", "stillness, relaxing, nothing happening"),
    "cute vs serious": ("adorable, chibi, kawaii", "grim, mature, realistic tone"),
    "fantasy vs everyday": ("dragons, magic, spaceships, other worlds", "a normal street, kitchen or office"),
    "people vs no people": ("a human face or figure is the main subject", "objects or landscape only, nobody in it"),
    "painterly vs crisp": ("visible paint strokes on canvas", "smooth, sharp, polished digital render"),
}


def check_paraphrase():
    store, pop, _ = load()
    from clip_model import Clip
    clip = Clip()
    rng = np.random.default_rng(0)
    E = store.emb[rng.choice(len(store.emb), 4000, replace=False)]
    log("\n== D. Paraphrase test with independently written wordings (corr of image scores, 4000 images)")
    for ax, (p, n) in STARTER.items():
        a = clip.embed_texts([p, n]); b = clip.embed_texts(list(ALT[ax]))
        r = np.corrcoef(E @ (a[0] - a[1]), E @ (b[0] - b[1]))[0, 1]
        log(f"   {ax:26s} r = {r:+.2f}  {'ok' if r >= 0.7 else 'WORDING-SENSITIVE' if r >= 0.4 else 'UNRELIABLE'}")


# ------------------------------------------------------------------ E
def check_app():
    import subprocess
    import urllib.request
    store, pop, pairs = load()
    log("\n== E1. Rating app over HTTP with a simulated user (oracle = a stand-in user's fitted taste)")
    u, df = targets(pairs)[-1]
    X, y = store.pair_diffs(df)
    oracle = TasteProfile.fit(u, X / pop.scale, y, pop.w, pop.lam_prior)
    user = "_simtest"
    for p in (f"data/feedback/{user}.jsonl", f"data/profiles/{user}.npz"):
        if os.path.exists(p):
            os.remove(p)
    srv = subprocess.Popen([sys.executable, "-W", "ignore", "src/rate_app.py", "--user", user, "--port", "8798"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(60):
            try:
                urllib.request.urlopen("http://127.0.0.1:8798/", timeout=2); break
            except Exception:
                time.sleep(1)
        import re
        rng = np.random.default_rng(0)
        for i in range(40):
            page = urllib.request.urlopen("http://127.0.0.1:8798/").read().decode()
            pid = re.search(r"pid:'([0-9a-f]+)'", page).group(1)
            L, R = re.findall(r'src="/img/([^"]+)"', page)[:2]
            cap = re.search(r'class="prompt">([^<]*)', page).group(1)
            import html as _h
            x = store.features([L, R], [_h.unescape(cap)] * 2) / pop.scale
            side = "left" if rng.random() < sigmoid((x[0] - x[1]) @ oracle.mean) else "right"
            req = urllib.request.Request("http://127.0.0.1:8798/choice", method="POST",
                                         data=json.dumps({"pid": pid, "side": side, "response_ms": 1500}).encode(),
                                         headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req)
        fb = [json.loads(l) for l in open(f"data/feedback/{user}.jsonl")]
        ok_fields = all({"caption", "left", "right", "chosen", "p_pred", "response_ms"} <= set(r) for r in fb)
        prof = TasteProfile.load(f"data/profiles/{user}.npz")
        log(f"   40 choices posted -> {len(fb)} feedback rows, all fields present: {ok_fields}, "
            f"profile saved with n_pairs={prof.n_pairs}")
        prof_page = urllib.request.urlopen("http://127.0.0.1:8798/profile?given=photorealistic").read().decode()
        log(f"   /profile with context renders: {'Based on' in prof_page}")
    finally:
        srv.terminate()
        for p in (f"data/feedback/{user}.jsonl", f"data/profiles/{user}.npz"):
            if os.path.exists(p):
                os.remove(p)

    log("\n== E2. Active vs random pair selection (simulated users, 3 oracles x 3 seeds, held-out oracle agreement)")
    import random as pyrandom
    from rate_app import State
    st = State.__new__(State)
    st.store, st.pop = store, pop
    pool = pairs[pairs.role == "global"]
    st.pool = pool[pool.caption.str.count(r"[A-Za-z]{2,}") >= 3].reset_index(drop=True)
    rng_eval = np.random.default_rng(42)
    ev = st.pool.sample(2000, random_state=1)
    Xev = store.pair_diffs(ev)[0] / pop.scale
    checkpoints = (10, 25, 50, 100)
    results = {m: {k: [] for k in checkpoints} for m in ("active", "random")}
    # oracles fit WITHOUT the population prior so they have distinct taste; otherwise
    # the prior alone already agrees with them and the comparison says nothing
    def make_oracle(df):
        X, y = store.pair_diffs(df)
        return TasteProfile.fit("oracle", X / pop.scale, y, np.zeros(FEATS), 300.0)

    for oi, (u, df) in enumerate(targets(pairs)[:3]):
        oracle = make_oracle(df)
        truth = (Xev @ oracle.mean) > 0
        for mode in ("active", "random"):
            for seed in range(3):
                pyrandom.seed(seed); rng = np.random.default_rng(seed)
                st.seen, st.pending = set(), {}
                st.prof = pop.cold_profile("sim")
                Xs_hist, y_hist = [], []
                orig = pyrandom.random
                if mode == "random":
                    pyrandom.random = lambda: 0.99  # forces the random branch in next_pair
                try:
                    for n in range(1, max(checkpoints) + 1):
                        pid, p = st.next_pair()
                        x = store.pair_diffs(pd.DataFrame({"caption": [p["caption"]], "image_0_uid": [p["left"]],
                                                           "image_1_uid": [p["right"]], "label_0": [1.0]}))[0] / pop.scale
                        Xs_hist.append(x[0]); y_hist.append(float(rng.random() < sigmoid(x[0] @ oracle.mean)))
                        st.seen |= {(p["left"], p["right"]), (p["right"], p["left"])}
                        st.prof = TasteProfile.fit("sim", np.array(Xs_hist), np.array(y_hist), pop.w, pop.lam_prior)
                        if n in checkpoints:
                            results[mode][n].append(float(((Xev @ st.prof.mean) > 0).__eq__(truth).mean()))
                finally:
                    pyrandom.random = orig
    base = []
    for u, df in targets(pairs)[:3]:
        oracle = make_oracle(df)
        base.append(float((((Xev @ pop.w) > 0) == ((Xev @ oracle.mean) > 0)).mean()))
    log(f"   agreement with oracle before any ratings (population prior): {np.mean(base):.3f}")
    for n in checkpoints:
        log(f"   after {n:3d} ratings: active {np.mean(results['active'][n]):.3f}   random {np.mean(results['random'][n]):.3f}")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    log(f"\n######## verify {what}  {time.strftime('%Y-%m-%d %H:%M')}")
    fns = dict(splits=check_splits, rank=check_rank, readout=check_readout, paraphrase=check_paraphrase, app=check_app)
    for k, f in fns.items():
        if what in (k, "all"):
            f()
