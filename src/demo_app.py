"""Visual demo of the whole project, one local page with tabs.

  python src/demo_app.py                  # http://localhost:8770
  python src/rate_app.py --user abc     # http://localhost:8765  (live learning + profile; linked from the demo)

Tabs: Why this image (ranking + reasons + visual layers) · Same prompt, different people ·
Evidence (measured results) · Limits. Everything shown is computed from the real data/models.
"""
from __future__ import annotations

import argparse
import html
import io
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, quote, urlparse

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from conditional import Conditioner  # noqa: E402
from rerank import reasons, reference_pool  # noqa: E402
from taste import EMB_DIM, Store, TasteProfile, accuracy, sigmoid  # noqa: E402
from user_profile import Population  # noqa: E402

E = html.escape


# ------------------------------------------------------------------ state
class Demo:
    def __init__(self):
        from clip_model import Clip
        self.store, self.pop = Store(), Population()
        self.clip = Clip()
        self.cond = Conditioner(self.store, self.pop, self.clip)
        self.ref = reference_pool(self.store, self.pop, n=600)
        pairs = pd.read_parquet("data/pairs_ready.parquet")
        self.pairs = pairs
        tgt = pairs[pairs.role == "target"]
        self.users = [str(u) for u in tgt.user_id.value_counts().index]
        self.prof = {}
        for u, d in tgt.groupby("user_id"):
            X, y = self.store.pair_diffs(d)
            self.prof[str(u)] = TasteProfile.fit(str(u), X / self.pop.scale, y, self.pop.w, self.pop.lam_prior)
        self.prof["average user"] = TasteProfile("average user", self.pop.w, np.eye(len(self.pop.w)) / self.pop.lam_prior, 0)
        # how far each user's taste is from the population (used to pick contrasting users)
        self.distinct = sorted(self.users, key=lambda u: -np.linalg.norm(self.prof[u].mean - self.pop.w))
        # prompts with >= 5 images, real words, images on disk
        imgs = {}
        for _, r in pairs.iterrows():
            for u in (r.image_0_uid, r.image_1_uid):
                imgs.setdefault(r.caption, set()).add(u)
        good = [(c, sorted(v)) for c, v in imgs.items()
                if len(v) >= 5 and len(c) < 160 and sum(w.isalpha() for w in c.split()) >= 3
                and all(os.path.exists(f"data/images/{u}.jpg") for u in v)]
        good.sort(key=lambda t: -len(t[1]))
        self.prompts = good[:300]
        self._depth = None
        self.meas = pd.read_parquet("data/measured_color.parquet").set_index("uid")
        self.evidence = self._evidence()

    # -------- ranking for one user / prompt
    def rank(self, user, pidx, limit=8):
        cap, uids = self.prompts[pidx]
        uids = uids[:limit]
        prof = self.prof[user]
        X = self.store.features(uids, [cap] * len(uids)) / self.pop.scale
        Xt = X.copy(); Xt[:, EMB_DIM] = 0
        p, lo, hi = prof.p_like(Xt, self.ref, n_samples=150)
        emb = self.store.emb[[self.store.idx[u] for u in uids]]
        why = reasons(prof, self.cond, emb)
        order = np.argsort(-(X @ prof.mean))
        return [dict(uid=uids[i], p=p[i], lo=lo[i], hi=hi[i], why=why[i]) for i in order]

    # -------- visual layers
    def depth(self, uid):
        import torch
        from PIL import Image
        if self._depth is None:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
            name = "depth-anything/Depth-Anything-V2-Small-hf"
            dev = "mps" if torch.backends.mps.is_available() else "cpu"
            self._depth = (AutoImageProcessor.from_pretrained(name),
                           AutoModelForDepthEstimation.from_pretrained(name).to(dev).eval(), dev)
        proc, model, dev = self._depth
        im = Image.open(f"data/images/{uid}.jpg").convert("RGB")
        im.draft("RGB", (448, 448))
        with torch.no_grad():
            d = model(**proc(images=im, return_tensors="pt").to(dev)).predicted_depth[0].float().cpu().numpy()
        return (d - d.min()) / (np.ptp(d) + 1e-6)

    def layer_png(self, uid, kind):
        img = cv2.imread(f"data/images/{uid}.jpg")
        img = cv2.resize(img, (384, int(384 * img.shape[0] / img.shape[1])))
        if kind == "depth":
            d = cv2.resize(self.depth(uid), (img.shape[1], img.shape[0]))
            out = cv2.applyColorMap((d * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        elif kind == "subject":
            d = cv2.resize(self.depth(uid), (img.shape[1], img.shape[0]))
            thr, _ = cv2.threshold((d * 255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            fg = (d * 255 > thr)
            out = (img * 0.25).astype(np.uint8)
            out[fg] = img[fg]
            cnts, _ = cv2.findContours(fg.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, cnts, -1, (80, 200, 255), 2)
        elif kind == "edges":
            out = 255 - cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 100, 200)
            out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
        else:
            out = img
        return cv2.imencode(".png", out)[1].tobytes()

    def palette(self, uid, k=6):
        img = cv2.imread(f"data/images/{uid}.jpg", cv2.IMREAD_REDUCED_COLOR_4)
        px = img.reshape(-1, 3).astype(np.float32)
        _, lab, cen = cv2.kmeans(px, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0), 3,
                                 cv2.KMEANS_PP_CENTERS)
        frac = np.bincount(lab.ravel(), minlength=k) / len(lab)
        return [(f"#{int(c[2]):02x}{int(c[1]):02x}{int(c[0]):02x}", f) for c, f in sorted(zip(cen, frac), key=lambda t: -t[1])]

    # -------- evidence (recomputed from data so the numbers are live, not copied)
    def _evidence(self):
        ev = json.load(open("reports/eval.json"))
        tgt = self.pairs[self.pairs.role == "target"]
        y_all, p_all = [], []
        for u, d in tgt.groupby("user_id"):
            d = d.sort_values("created_at"); c = int(0.7 * len(d))
            Xtr, ytr = self.store.pair_diffs(d.iloc[:c]); Xte, yte = self.store.pair_diffs(d.iloc[c:])
            prof = TasteProfile.fit(str(u), Xtr / self.pop.scale, ytr, self.pop.w, self.pop.lam_prior)
            y_all.append(yte); p_all.append(prof.p_prefer(Xte / self.pop.scale))
        y, p = np.concatenate(y_all), np.concatenate(p_all)
        conf = np.where(p >= .5, p, 1 - p); hit = np.where(p >= .5, y, 1 - y)
        bins = []
        for lo in np.arange(.5, 1.0, .05):
            m = (conf >= lo) & (conf < lo + .05)
            if m.sum() >= 15:
                bins.append((float(conf[m].mean()), float(hit[m].mean()), int(m.sum())))
        axes = pd.read_csv("reports/axes_check.csv") if os.path.exists("reports/axes_check.csv") else None
        return dict(summary=ev["summary"], curve=ev["curve"], gap=ev["gap"], calib=bins, n_test=len(y),
                    acc=accuracy(y, p), axes=axes)


# ------------------------------------------------------------------ html
CSS = """
:root{--bg:#f6f5f2;--fg:#1d1d1b;--mute:#6b6a66;--card:#fff;--line:#dedcd6;--acc:#2f5bd3;--good:#1f8a4c;--bad:#c2452d;--on:#fff}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--fg:#ecebe7;--mute:#9b9a95;--card:#212120;--line:#353532;--acc:#7c9cff;--good:#4cc07f;--bad:#ef7a62;--on:#111}}
*{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.45 -apple-system,system-ui,sans-serif}
header{padding:14px 16px;border-bottom:1px solid var(--line)} header h1{margin:0 0 2px;font-size:19px}
nav{display:flex;gap:6px;flex-wrap:wrap;padding:10px 16px;border-bottom:1px solid var(--line)}
nav a{padding:6px 12px;border-radius:999px;text-decoration:none;color:var(--fg);border:1px solid var(--line);background:var(--card);font-size:14px}
nav a.on{background:var(--acc);color:var(--on);border-color:var(--acc)} main{max-width:1240px;margin:0 auto;padding:16px}
.mute{color:var(--mute)} a{color:var(--acc)} select,button{font:inherit;padding:7px 10px;border-radius:8px;border:1px solid var(--line);background:var(--card);color:var(--fg)}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:8px} .card img{width:100%;border-radius:6px;display:block}
.rank{font-weight:700;font-size:13px} .p{font-size:20px;font-weight:700} .why{font-size:12.5px;margin-top:4px}
.why .pos{color:var(--good)} .why .neg{color:var(--bad)}
.cols{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px} @media (max-width:900px){.cols{grid-template-columns:repeat(2,minmax(0,1fr))}}
.col h3{margin:4px 0 8px;font-size:15px} .mini{display:flex;gap:8px;align-items:center;margin-bottom:6px;background:var(--card);border:1px solid var(--line);border-radius:8px;padding:4px}
.mini img{width:64px;height:64px;object-fit:cover;border-radius:5px}
.layers{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px} @media (max-width:800px){.layers{grid-template-columns:repeat(2,minmax(0,1fr))}}
.layers figure{margin:0} .layers img{width:100%;border-radius:6px} figcaption{font-size:12.5px;margin-top:3px}
.sw{display:flex;height:34px;border-radius:6px;overflow:hidden;margin-top:6px}
table{border-collapse:collapse} td,th{padding:5px 10px;border-bottom:1px solid var(--line);text-align:left;font-size:14px}
.panel{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px;margin-bottom:14px}
.two{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:14px} @media (max-width:860px){.two{grid-template-columns:1fr}}
svg text{fill:var(--fg);font:12px -apple-system,system-ui,sans-serif} .chip{display:inline-block;padding:2px 8px;border-radius:999px;font-size:12px;border:1px solid var(--line)}
"""

TABS = [("why", "Why this image?"), ("people", "Same prompt, different people"), ("evidence", "Can you trust it?"),
        ("limits", "What it can't do yet"), ("live", "Watch it learn ↗")]


def page(tab, body):
    nav = "".join(f'<a class="{"on" if k == tab else ""}" href="/?tab={k}">{v}</a>' for k, v in TABS)
    return (f'<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>Taste Engine Demo</title><style>{CSS}</style></head><body><header><h1>Taste Engine</h1>'
            f'<div class="mute">Learns each person\'s visual taste from their choices, predicts how likely they are to like '
            f'a new AI image, and says why.</div></header><nav>{nav}</nav><main>{body}</main></body></html>')


def why_html(reason: str):
    parts = reason.split("  ")
    return " ".join(f'<span class="{"pos" if r.startswith("+") else "neg"}">{E(r)}</span>' for r in parts if r.strip())


def selector(d: Demo, tab, user, pidx, users=True):
    uo = "".join(f'<option {"selected" if u == user else ""}>{E(u)}</option>' for u in d.distinct + ["average user"])
    po = "".join(f'<option value="{i}" {"selected" if i == pidx else ""}>{E(c[:90])} ({len(v)} images)</option>'
                 for i, (c, v) in enumerate(d.prompts))
    us = f'<label>User <select name="user">{uo}</select></label> ' if users else ""
    return (f'<form method="get" style="display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-bottom:14px">'
            f'<input type="hidden" name="tab" value="{tab}">{us}<label>Prompt <select name="p" style="max-width:min(640px,90vw)">{po}</select></label>'
            f'<button>Show</button></form>')


def tab_why(d: Demo, q):
    user = q.get("user", d.distinct[0]); pidx = int(q.get("p", 0)); focus = q.get("focus")
    ranked = d.rank(user, pidx)
    cap = d.prompts[pidx][0]
    n = d.prof[user].n_pairs
    head = (f'<p><b>Prompt:</b> {E(cap)}<br><span class="mute">Ranked for user <b>{E(user)}</b> '
            f'(profile from {n} of their real choices). P(like) = chance they pick this over a typical generated image; '
            f'range = 80% interval. Click an image to see its layers.</span></p>')
    layers = ""
    if focus:
        m = d.meas.loc[focus] if focus in d.meas.index else None
        pal = d.palette(focus)
        sw = "".join(f'<div style="background:{c};flex:{f:.3f}" title="{c} {f:.0%}"></div>' for c, f in pal)
        stats = "" if m is None else (f'brightness {m.m_brightness:.2f} · saturation {m.m_saturation:.2f} · '
                                      f'contrast {m.m_contrast:.2f} · edge density {m.m_edge_density:.3f} · sharpness {m.m_sharpness:.1f}')
        layers = (f'<div class="panel"><b>Visual layers</b> <span class="mute">· measured, not guessed from words</span>'
                  f'<div class="layers" style="margin-top:10px">'
                  f'<figure><img src="/layer/{focus}/image"><figcaption>original</figcaption></figure>'
                  f'<figure><img src="/layer/{focus}/depth"><figcaption>depth map (bright = near)</figcaption></figure>'
                  f'<figure><img src="/layer/{focus}/subject"><figcaption>subject mask (from depth)</figcaption></figure>'
                  f'<figure><img src="/layer/{focus}/edges"><figcaption>edges (detail / clutter)</figcaption></figure></div>'
                  f'<div class="sw">{sw}</div><div class="mute" style="font-size:13px;margin-top:6px">colour palette · {stats}</div></div>')
    cards = "".join(
        f'<a class="card" style="text-decoration:none;color:inherit" href="/?tab=why&user={quote(user)}&p={pidx}&focus={r["uid"]}">'
        f'<img src="/img/{r["uid"]}" loading="lazy"><div class="rank">#{i + 1}</div>'
        f'<div><span class="p">{r["p"]:.0%}</span> <span class="mute">[{r["lo"]:.0%}–{r["hi"]:.0%}]</span></div>'
        f'<div class="why">{why_html(r["why"])}</div></a>' for i, r in enumerate(ranked))
    return page("why", selector(d, "why", user, pidx) + head + layers + f'<div class="grid">{cards}</div>')


def tab_people(d: Demo, q):
    pidx = int(q.get("p", 0))
    cap = d.prompts[pidx][0]
    picks = d.distinct[:3] + ["average user"]
    cols = []
    tops = []
    for u in picks:
        r = d.rank(u, pidx, limit=8)
        tops.append(r[0]["uid"])
        items = "".join(f'<div class="mini"><b class="mute">{i + 1}</b><img src="/img/{x["uid"]}"><div>'
                        f'<b>{x["p"]:.0%}</b><div class="why">{why_html(x["why"])}</div></div></div>' for i, x in enumerate(r[:5]))
        label = "average user (population)" if u == "average user" else f"user {u}"
        cols.append(f'<div class="col"><h3>{E(label)}</h3>{items}</div>')
    agree = len(set(tops))
    note = (f'<p class="mute">Top pick differs across these {len(picks)} columns: {agree} different top images. '
            'Users shown are the three whose learned taste is furthest from the average. Differences are real but '
            'modest; see “What it can’t do yet”.</p>')
    return page("people", selector(d, "people", None, pidx, users=False) + f'<p><b>Prompt:</b> {E(cap)}</p>' + note
                + f'<div class="cols">{"".join(cols)}</div>')


def bar_chart(items, maxv=0.8, minv=0.4, w=560, unit="%"):
    h = 28 * len(items) + 20
    rows = []
    x0 = 190
    for i, (label, v, hl) in enumerate(items):
        bw = (v - minv) / (maxv - minv) * (w - x0 - 60)
        y = 10 + i * 28
        rows.append(f'<text x="0" y="{y + 15}">{E(label)}</text><rect x="{x0}" y="{y + 3}" width="{max(bw, 1):.1f}" height="18" rx="3" '
                    f'fill="var(--{"acc" if hl else "line"})"/><text x="{x0 + bw + 6:.1f}" y="{y + 16}">{v * 100:.1f}{unit}</text>')
    x50 = x0 + (0.5 - minv) / (maxv - minv) * (w - x0 - 60)
    rows.append(f'<line x1="{x50:.1f}" x2="{x50:.1f}" y1="4" y2="{h - 4}" stroke="var(--mute)" stroke-dasharray="3 3"/>')
    return f'<svg viewBox="0 0 {w} {h}" width="100%" style="max-width:{w}px">{"".join(rows)}</svg>'


def calib_chart(bins, s=260):
    pad = 34
    sc = lambda v: pad + (v - .5) / .5 * (s - 2 * pad)  # noqa: E731
    pts = "".join(f'<circle cx="{sc(c):.1f}" cy="{s - sc(h):.1f}" r="{3 + min(n, 400) / 80:.1f}" fill="var(--acc)"/>' for c, h, n in bins)
    return (f'<svg viewBox="0 0 {s} {s}" width="100%" style="max-width:{s}px"><line x1="{sc(.5)}" y1="{s - sc(.5)}" x2="{sc(1)}" '
            f'y2="{s - sc(1)}" stroke="var(--mute)" stroke-dasharray="4 4"/>{pts}<text x="{pad}" y="{s - 6}">predicted confidence →</text>'
            f'<text x="4" y="14">actually right ↑</text><text x="{sc(.5) - 8}" y="{s - pad + 14}">50%</text><text x="{sc(1) - 26}" y="{s - pad + 14}">100%</text></svg>')


def tab_evidence(d: Demo, q):
    ev = d.evidence
    s = {r["method"]: r["pooled_acc"] for r in ev["summary"]}
    bars = bar_chart([("coin flip", s["random"], False), ("“prettiness” score (LAION)", s["laion_aesthetic"], False),
                      ("prompt match only", s["prompt_alignment"], False), ("personal only, no prior", s["personal_only"], False),
                      ("average user (population)", s["population"], False), ("personal + population prior", s["personal+prior"], True)])
    cv = pd.DataFrame(ev["curve"])
    cv = cv[cv.method == "personal+prior"].groupby("n").acc.mean()
    curve = bar_chart([(f"{n} of their ratings", v, n == cv.index.max()) for n, v in cv.items() if n <= 200], maxv=.75, minv=.45)
    axes = ""
    if ev["axes"] is not None:
        a = ev["axes"]
        rows = "".join(f'<tr><td>{E(r.task)}</td><td>{r.clip_zeroshot:.2f}</td><td>{r.clip_probe:.2f}</td>'
                       f'<td>{"" if pd.isna(r.get("measured", np.nan)) else f"{r.measured:.2f}"}</td></tr>'
                       for _, r in a.iterrows() if not pd.isna(r.clip_probe))
        axes = (f'<div class="panel"><b>Which attributes can be trusted?</b> <span class="mute">AUC vs. labels from a local '
                f'vision-language model (0.5 = chance, ≥0.85 usable). Its labels for angle, brightness, saturation and clutter '
                f'were too uniform to test.</span><table style="margin-top:8px"><tr><th>attribute</th><th>CLIP from words</th>'
                f'<th>CLIP + trained tagger</th><th>measured layers</th></tr>{rows}</table>'
                f'<p class="mute">Pixel truth: CLIP’s “dark vs bright” correlates only 0.30 with real brightness, so brightness, '
                f'saturation and contrast come from pixel measurements instead.</p></div>')
    g = ev["gap"]
    return page("evidence", f"""
<div class="two"><div class="panel"><b>Predicting which of two images a person picks</b>
<div class="mute">{ev['n_test']} later choices from 10 real users, predicted from their earlier ones</div>{bars}
<p class="mute">Personal over population: {g['mean'] * 100:+.1f} points (95% range {g['lo'] * 100:+.1f} to {g['hi'] * 100:+.1f}). A beauty score does
<b>worse than a coin flip</b>: pretty ≠ preferred.</p></div>
<div class="panel"><b>When it says 70%, is it right 70% of the time?</b><div class="mute">Each dot = a group of predictions; on the dashed line = honest confidence</div>
{calib_chart(ev['calib'])}<p class="mute">Calibration error 0.029: the confidence numbers can be taken at face value.</p></div></div>
<div class="two"><div class="panel"><b>Ranking: is their favourite on top?</b><p>Among 3+ images for one prompt, the model's top pick is the
user's actual favourite <b>38–40%</b> of the time vs <b>21%</b> by chance (154 prompt groups).</p></div>
<div class="panel"><b>Learning curve</b><div class="mute">accuracy vs. number of the user's own ratings</div>{curve}</div></div>
{axes}""")


def tab_limits(d: Demo, q):
    items = [
        ("Personal taste adds only a little so far", "+1.5 points over the average user (not yet statistically certain). These Pick-a-Pic users choose fast; their early choices only weakly predict later ones. A deliberate rater (user abc) may be more consistent: unknown until they rate more."),
        ("Low-rank personalisation did not help", "Tested: 67.4% vs 70.0% for the current model. Kept the current model."),
        ("Context-specific taste is not established", "“In photoreal images he prefers…” readouts exist, but a context-aware model did not predict better than a plain one. A neural model (option D) is parked until this test passes."),
        ("Some attributes are unverified", "Camera angle and clutter: the local vision model's labels were too uniform to check against, and CLIP failed a paraphrase test on angle (0.22)."),
        ("Explanations share credit between related attributes", "dark / moody / low saturation move together in real images, so reasons can name a neighbour."),
        ("Implicit feedback is recorded but unused", "Response time is logged with every choice; it does not yet weight the model."),
    ]
    body = "".join(f'<div class="panel"><b>{E(t)}</b><div class="mute">{E(x)}</div></div>' for t, x in items)
    return page("limits", body)


def tab_live(d: Demo, q):
    return page("live", '<div class="panel"><b>Watch it learn: live rating app</b><p>Pick between two images from the same prompt. '
                'Before each click the model\'s prediction is logged, so the running “predicted your choice” number is honest. '
                'Your taste sliders and the context box (“in photorealistic images you prefer…”) update after every choice.</p>'
                '<p><a href="http://localhost:8765/" target="_blank">Open the rater →</a> &nbsp; '
                '<a href="http://localhost:8765/profile" target="_blank">Open the taste profile →</a></p>'
                '<p class="mute">Needs: <code>.venv/bin/python src/rate_app.py --user abc</code></p></div>')


def handler(d: Demo):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def send(self, b, ctype="text/html; charset=utf-8", code=200):
            b = b if isinstance(b, bytes) else b.encode()
            self.send_response(code); self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

        def do_GET(self):
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            try:
                if u.path.startswith("/img/"):
                    fp = f"data/images/{os.path.basename(u.path[5:])}.jpg"
                    return self.send(open(fp, "rb").read(), "image/jpeg")
                if u.path.startswith("/layer/"):
                    _, _, uid, kind = u.path.split("/")
                    return self.send(d.layer_png(os.path.basename(uid), kind), "image/png")
                tab = q.get("tab", "why")
                fn = dict(why=tab_why, people=tab_people, evidence=tab_evidence, limits=tab_limits, live=tab_live).get(tab, tab_why)
                self.send(fn(d, q))
            except Exception as e:  # show errors in the page instead of a dead socket during a demo
                import traceback
                self.send(f"<pre>{E(traceback.format_exc())}</pre>", code=500)
    return H


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8770)
    a = ap.parse_args()
    d = Demo()
    print(f"demo ready: http://localhost:{a.port}  ({len(d.prompts)} prompts, users by distinctness: {d.distinct[:3]})", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), handler(d)).serve_forever()
