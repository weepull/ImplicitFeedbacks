"""Do the axes measure what their names say? Compare against real labels.

Per attribute (binary task, e.g. close-up vs wide), AUC of each signal against labels:
  clip_zeroshot : the CLIP phrase axis from axes.py, sign fixed in advance (no training)
  measured      : logistic regression on measured layers (colour/edge/depth), 5-fold CV
  clip_probe    : logistic regression on the 768-d CLIP embedding, trained on VLM labels, 5-fold CV
Labels: VLM (Qwen2.5-VL-3B, ~2000 imgs) and, if present, blind human labels (~100 imgs).
Also: VLM vs human agreement, i.e. how far the VLM itself can be trusted.
AUC 0.5 = no signal, 1.0 = perfect. Rule of thumb: >=0.85 usable, 0.7-0.85 rough, <0.7 don't trust.

  python src/check_axes.py     -> prints table, writes reports/axes_check.csv
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))

# attribute task: (label field, positive values, negative values, CLIP axis, sign, measured columns)
TASKS = {
    "photo vs drawn":      ("style", ["photo"], ["illustration", "anime", "painting"], "photoreal vs illustrated", +1, []),
    "anime vs other":      ("style", ["anime"], ["photo", "illustration", "painting", "3d_render"], "anime vs western", +1, []),
    "close-up vs wide":    ("shot", ["close_up"], ["wide"], "close-up vs wide", +1,
                            ["d_fg_frac", "d_range", "d_median", "d_centre_minus_edge", "m_edge_density"]),
    "low vs other angle":  ("angle", ["low"], ["eye_level", "high"], "low vs eye-level angle", +1,
                            ["d_vgrad", "d_fg_cy", "d_fg_frac"]),
    "dramatic vs flat":    ("lighting", ["dramatic"], ["flat"], "dramatic vs flat light", +1,
                            ["m_contrast", "m_dark_frac", "m_brightness"]),
    "dark vs bright":      ("brightness", ["dark"], ["bright"], "dark vs bright", +1,
                            ["m_brightness", "m_dark_frac", "m_contrast"]),
    "busy vs minimal":     ("clutter", ["busy"], ["minimal"], "busy vs minimal", +1,
                            ["m_edge_density", "m_palette_entropy", "d_range"]),
    "action vs calm":      ("energy", ["action"], ["calm"], "action vs calm", +1, []),
    "people vs none":      ("people", ["yes"], ["no"], "people vs no people", +1, []),
    "fantasy vs everyday": ("setting", ["fantasy"], ["everyday"], "fantasy vs everyday", +1, []),
    "cute vs serious":     ("mood", ["cute"], ["serious"], "cute vs serious", +1, []),
    "vivid vs muted":      ("saturation", ["vivid"], ["muted"], "saturated vs muted", +1,
                            ["m_saturation", "m_colourfulness", "m_palette_entropy"]),
}


def binarize(df, field, pos, neg):
    v = df[field]
    keep = v.isin(pos + neg)
    return keep.values, v[keep].isin(pos).astype(int).values


def cv_auc(X, y, C=1.0):
    if len(np.unique(y)) < 2 or min(np.bincount(y)) < 10:
        return np.nan
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))
    p = cross_val_predict(clf, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
    return roc_auc_score(y, p)


def main():
    from axes import AxisRegistry
    from clip_model import Clip
    z = np.load("data/embeddings.npz")
    idx = {u: i for i, u in enumerate(z["uids"])}
    E = z["emb"].astype(np.float32)
    reg = AxisRegistry(Clip(), E)

    vlm = pd.DataFrame([json.loads(l) for l in open("data/labels/vlm.jsonl")])
    vlm = vlm[vlm.uid.isin(idx)].drop_duplicates("uid").reset_index(drop=True)
    meas = None
    for part in ("color", "depth"):
        p = f"data/measured_{part}.parquet"
        if os.path.exists(p):
            m = pd.read_parquet(p)
            meas = m if meas is None else meas.merge(m, on="uid", how="outer")
    human = None
    if os.path.exists("data/labels/human.jsonl"):
        human = pd.DataFrame([json.loads(l) for l in open("data/labels/human.jsonl")]).drop_duplicates("uid", keep="last")

    print(f"VLM-labelled images: {len(vlm)}   human-labelled: {0 if human is None else len(human)}   "
          f"measured layers: {'none' if meas is None else ', '.join(p for p in ('color', 'depth') if os.path.exists(f'data/measured_{p}.parquet'))}")

    # value distribution sanity check: a 3B model may default to 'safe' answers
    print("\nVLM label distribution (watch for one value dominating):")
    for f in ("style", "shot", "angle", "lighting", "brightness", "clutter", "energy", "mood", "saturation"):
        print(f"  {f:11s} " + "  ".join(f"{k}={v:.0%}" for k, v in vlm[f].value_counts(normalize=True).items()))

    rows = []
    for name, (field, pos, neg, axis, sign, mcols) in TASKS.items():
        keep, y = binarize(vlm, field, pos, neg)
        sub = vlm[keep]
        Ei = E[[idx[u] for u in sub.uid]]
        r = dict(task=name, n_vlm=len(y), pos_rate=y.mean() if len(y) else np.nan)
        r["clip_zeroshot"] = roc_auc_score(y, sign * reg.z(axis, Ei)) if len(np.unique(y)) == 2 else np.nan
        r["clip_probe"] = cv_auc(Ei, y, C=0.5)
        if meas is not None and mcols:
            mm = sub[["uid"]].merge(meas, on="uid", how="left")
            cols = [c for c in mcols if c in mm.columns]
            ok = mm[cols].notna().all(1).values if cols else np.zeros(len(mm), bool)
            r["measured"] = cv_auc(mm.loc[ok, cols].values, y[ok]) if ok.sum() > 50 else np.nan
            if ok.sum() > 50:
                r["clip+measured"] = cv_auc(np.hstack([Ei[ok], mm.loc[ok, cols].values * 5]), y[ok], C=0.5)
        if human is not None and field in human:
            hm = human.merge(vlm[["uid", field]], on="uid", suffixes=("_h", "_v"))
            hm = hm[(hm[f"{field}_h"] != "unsure") & hm[f"{field}_v"].notna()]
            r["vlm_vs_human_agree"] = (hm[f"{field}_h"] == hm[f"{field}_v"]).mean() if len(hm) else np.nan
            hk, hy = binarize(human.rename(columns={}), field, pos, neg)
            hs = human[hk]
            if len(np.unique(hy)) == 2 and len(hy) >= 10:
                Eh = E[[idx[u] for u in hs.uid]]
                r["clip_zeroshot_vs_human"] = roc_auc_score(hy, sign * reg.z(axis, Eh))
                r["n_human"] = len(hy)
        rows.append(r)

    df = pd.DataFrame(rows)
    os.makedirs("reports", exist_ok=True)
    df.to_csv("reports/axes_check.csv", index=False)
    pd.set_option("display.width", 200)
    print("\nAUC against VLM labels (0.5 = no signal; >=0.85 usable, 0.7-0.85 rough, <0.7 don't trust)")
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
