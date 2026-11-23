"""Local rating app: a user picks between two images generated from the same prompt.

  python src/rate_app.py --user abc            # then open http://localhost:8765

Every choice is appended to data/feedback/<user>.jsonl and the profile is refit
immediately. Pairs are chosen by active learning (75%: the pair the current posterior
is most uncertain about; 25%: random, to avoid only ever seeing edge cases).
Before each reveal the model's prediction is logged (p_pred), so the running
"predicted-before-you-chose" accuracy on /profile is an honest held-out number.
Response time is logged as an implicit-feedback signal (not yet used in the fit).
"""
from __future__ import annotations

import argparse
import html
import json
import os
import random
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from taste import Store, sigmoid  # noqa: E402
from user_profile import (Population, describe, feedback_path, fit_from_feedback,  # noqa: E402
                          load_feedback, profile_path)

IMG_DIR = "data/images"


class State:
    def __init__(self, user):
        self.user = user
        self.store, self.pop = Store(), Population()
        pairs = pd.read_parquet("data/pairs_ready.parquet")
        pairs = pairs[pairs.role == "global"]
        ok = pairs.image_0_uid.map(lambda u: os.path.exists(f"{IMG_DIR}/{u}.jpg")) & \
            pairs.image_1_uid.map(lambda u: os.path.exists(f"{IMG_DIR}/{u}.jpg"))
        real_prompt = pairs.caption.str.count(r"[A-Za-z]{2,}") >= 3  # Pick-a-Pic has junk like "Gdhjyt"
        self.pool = pairs[ok & real_prompt].reset_index(drop=True)
        fb = load_feedback(user)
        self.seen = set(zip(fb.left, fb.right)) | set(zip(fb.right, fb.left)) if len(fb) else set()
        self.prof = fit_from_feedback(user, self.store, self.pop)
        self.pending: dict[str, dict] = {}
        self.lock = threading.Lock()
        self._clip = None
        os.makedirs("data/feedback", exist_ok=True)

    @property
    def clip(self):
        if self._clip is None:
            from clip_model import Clip
            self._clip = Clip()
        return self._clip

    @property
    def cond(self):
        if getattr(self, "_cond", None) is None:
            from conditional import Conditioner
            self._cond = Conditioner(self.store, self.pop, self.clip)
        return self._cond

    def next_pair(self):
        cand = self.pool.sample(min(400, len(self.pool)))
        cand = cand[[(a, b) not in self.seen for a, b in zip(cand.image_0_uid, cand.image_1_uid)]]
        X, _ = self.store.pair_diffs(cand)
        Xs = X / self.pop.scale
        if random.random() < 0.75:
            var = np.einsum("ij,jk,ik->i", Xs, self.prof.cov, Xs)
            row = cand.iloc[int(np.argmax(var))]
        else:
            row = cand.iloc[random.randrange(len(cand))]
        left, right = row.image_0_uid, row.image_1_uid
        if random.random() < 0.5:  # randomise screen side against position bias
            left, right = right, left
        x = self.store.pair_diffs(pd.DataFrame({"caption": [row.caption], "image_0_uid": [left],
                                                "image_1_uid": [right], "label_0": [1.0]}))[0] / self.pop.scale
        pid = uuid.uuid4().hex[:12]
        self.pending[pid] = dict(caption=row.caption, left=left, right=right,
                                 p_pred=float(self.prof.p_prefer(x)[0]))
        return pid, self.pending[pid]

    def record(self, pid, side, response_ms):
        with self.lock:
            p = self.pending.pop(pid, None)
            if p is None:
                return
            chosen = {"left": p["left"], "right": p["right"]}.get(side)
            rec = dict(caption=p["caption"], left=p["left"], right=p["right"], chosen=chosen,
                       response_ms=response_ms, p_pred=p["p_pred"], ts=time.time())
            with open(feedback_path(self.user), "a") as f:
                f.write(json.dumps(rec) + "\n")
            self.seen |= {(p["left"], p["right"]), (p["right"], p["left"])}
            if chosen is not None:
                self.prof = fit_from_feedback(self.user, self.store, self.pop)
                self.prof.save(profile_path(self.user))

    def stats(self):
        fb = load_feedback(self.user)
        fb = fb[fb.chosen.notna()] if len(fb) else fb
        if len(fb) == 0:
            return 0, None, None
        hit = ((fb.p_pred > 0.5) == (fb.chosen == fb.left)).values
        return len(fb), hit.mean(), hit[-50:].mean()


PAGE = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Taste Rater</title><style>
:root{--bg:#f6f5f2;--fg:#1d1d1b;--mute:#6b6a66;--card:#fff;--line:#dedcd6;--acc:#2f5bd3}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--fg:#ecebe7;--mute:#9b9a95;--card:#212120;--line:#353532;--acc:#7c9cff}}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.45 -apple-system,system-ui,sans-serif}
header{display:flex;justify-content:space-between;align-items:center;padding:12px 16px;border-bottom:1px solid var(--line)}
a{color:var(--acc)} .mute{color:var(--mute)} main{max-width:1200px;margin:0 auto;padding:16px}
.prompt{font-size:17px;margin:4px 0 16px} .row{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:8px;cursor:pointer;transition:transform .08s}
.card:hover{border-color:var(--acc)} .card:active{transform:scale(.99)} .card img{width:100%;border-radius:6px;display:block}
.keys{margin-top:14px;text-align:center} button{font:inherit;padding:8px 14px;border-radius:8px;border:1px solid var(--line);background:var(--card);color:var(--fg);cursor:pointer}
@media (max-width:640px){.row{gap:8px}}
</style></head><body><header><b>Taste Rater · __USER__</b><span class="mute">__STATS__ · <a href="/profile">profile</a></span></header>
<main><div class="mute">Prompt</div><div class="prompt">__CAPTION__</div>
<div class="row"><div class="card" onclick="go('left')"><img src="/img/__L__" onload="ready()"></div>
<div class="card" onclick="go('right')"><img src="/img/__R__" onload="ready()"></div></div>
<div class="keys mute">← left &nbsp;·&nbsp; → right &nbsp;·&nbsp; <button onclick="go('skip')">skip / can't decide (↓)</button></div></main>
<script>let n=0,t0=performance.now();function ready(){if(++n==2)t0=performance.now()}
let sent=false;function go(s){if(sent)return;sent=true;fetch('/choice',{method:'POST',headers:{'Content-Type':'application/json'},
body:JSON.stringify({pid:'__PID__',side:s,response_ms:Math.round(performance.now()-t0)})}).then(()=>location.reload())}
addEventListener('keydown',e=>{if(e.key=='ArrowLeft')go('left');if(e.key=='ArrowRight')go('right');if(e.key=='ArrowDown')go('skip')})</script></body></html>"""


def context_section(st: State, given: str | None) -> str:
    form = (f'<form method="get" action="/profile"><input name="given" placeholder="any context, e.g. photorealistic, '
            f'a night scene, portrait vs landscape" value="{html.escape(given or "")}" '
            f'style="width:min(560px,100%);font:inherit;padding:8px;border-radius:8px;border:1px solid var(--line);'
            f'background:var(--card);color:var(--fg)"> <button>Show</button></form>')
    if not given:
        return form
    from conditional import user_pairs
    pairs = user_pairs(st.user)
    if len(pairs) == 0:
        return form + '<p class="mute">No ratings yet. Rate some pairs first.</p>'
    df, n_eff = st.cond.readout(st.user, pairs, given)
    rows = "".join(
        f'<tr><td>{html.escape(r.axis)}</td><td><b>{r.effect:+.2f}</b> <span class="mute">[{r.lo:+.2f}, {r.hi:+.2f}]</span></td>'
        f'<td>{"clear" if r.clear else "<span class=mute>unclear</span>"}</td><td class="mute">{r.user_overall:+.2f}</td>'
        f'<td class="mute">{r.avg_user_in_context:+.2f}</td></tr>' for _, r in df.iterrows())
    return (form + f'<p class="mute">Based on ~{n_eff:.0f} of your ratings that match “{html.escape(given)}”. '
            'Effect = change in log-odds of picking an image 1 std further toward the first phrase.</p>'
            '<table><tr><th align=left>axis</th><th align=left>in this context</th><th></th>'
            '<th align=left>you overall</th><th align=left>avg user here</th></tr>' + rows + '</table>')


def profile_page(st: State, given: str | None = None):
    n, acc, acc50 = st.stats()
    d, extras = describe(st.prof, st.pop, st.clip)
    rng = np.random.default_rng(1)
    ii = rng.choice(len(st.store.uids), min(3000, len(st.store.uids)), replace=False)
    ii = np.array([i for i in ii if os.path.exists(f"{IMG_DIR}/{st.store.uids[i]}.jpg")])
    X = np.hstack([st.store.emb[ii], np.zeros((len(ii), 1)), st.store.aest[ii, None]]) / st.pop.scale
    s = X @ st.prof.mean
    order = np.argsort(-s)
    sel = np.concatenate([order[:12], order[-12:]])
    p, lo, hi = st.prof.p_like(X[sel], X[rng.choice(len(ii), 800, replace=False)])

    def grid(idx, ps):
        return "".join(f'<figure><img src="/img/{st.store.uids[ii[j]]}" loading="lazy"><figcaption>P(like) {pp:.2f} '
                       f'<span class="mute">[{l:.2f}–{h:.2f}]</span></figcaption></figure>'
                       for j, pp, l, h in zip(idx, *ps))
    axes = "".join(f'<tr><td>{html.escape(r.axis)}</td><td><div class="bar"><span style="left:{50 + min(r.lean, 0) * 50}%;'
                   f'width:{abs(r.lean) * 50}%"></span></div></td><td>{r.lean:+.2f}</td></tr>' for _, r in d.iterrows())
    accs = "no ratings yet" if n == 0 else f"{n} ratings · model predicted your choice {acc:.0%} of the time (last 50: {acc50:.0%})"
    return f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Taste Profile</title><style>
:root{{--bg:#f6f5f2;--fg:#1d1d1b;--mute:#6b6a66;--card:#fff;--line:#dedcd6;--acc:#2f5bd3}}
@media (prefers-color-scheme:dark){{:root{{--bg:#161615;--fg:#ecebe7;--mute:#9b9a95;--card:#212120;--line:#353532;--acc:#7c9cff}}}}
body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.45 -apple-system,system-ui,sans-serif}} main{{max-width:1200px;margin:0 auto;padding:16px}}
a{{color:var(--acc)}} .mute{{color:var(--mute)}} table{{border-collapse:collapse;width:100%;max-width:640px}} td{{padding:4px 8px}}
.bar{{position:relative;height:10px;background:var(--line);border-radius:5px;width:220px}} .bar span{{position:absolute;top:0;bottom:0;background:var(--acc);border-radius:5px}}
.g{{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:10px}} figure{{margin:0}} figure img{{width:100%;border-radius:6px}} figcaption{{font-size:12px}}
</style></head><body><main><p><a href="/">← keep rating</a></p><h2>{html.escape(st.user)}'s taste profile</h2><p class="mute">{accs}</p>
<h3>Taste in a context</h3>{context_section(st, given)}
<h3>Leanings vs. the average user</h3><p class="mute">Right = leans toward the first phrase. Read with care before ~50 ratings.</p><table>{axes}</table>
<h3>Most likely to like</h3><div class="g">{grid(sel[:12], (p[:12], lo[:12], hi[:12]))}</div>
<h3>Least likely to like</h3><div class="g">{grid(sel[12:], (p[12:], lo[12:], hi[12:]))}</div></main></body></html>"""


def make_handler(st: State):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def send(self, body, ctype="text/html; charset=utf-8", code=200):
            b = body if isinstance(body, bytes) else body.encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/":
                pid, p = st.next_pair()
                n, acc, _ = st.stats()
                stats = f"{n} rated" + (f" · {acc:.0%} predicted" if acc is not None else "")
                page = (PAGE.replace("__USER__", html.escape(st.user)).replace("__STATS__", stats)
                        .replace("__CAPTION__", html.escape(p["caption"])).replace("__L__", p["left"])
                        .replace("__R__", p["right"]).replace("__PID__", pid))
                self.send(page)
            elif path == "/profile":
                given = parse_qs(urlparse(self.path).query).get("given", [None])[0]
                self.send(profile_page(st, given))
            elif path.startswith("/img/"):
                uid = os.path.basename(path[5:])
                fp = f"{IMG_DIR}/{uid}.jpg"
                if os.path.exists(fp):
                    self.send(open(fp, "rb").read(), "image/jpeg")
                else:
                    self.send("not found", "text/plain", 404)
            else:
                self.send("not found", "text/plain", 404)

        def do_POST(self):
            if self.path == "/choice":
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                st.record(body["pid"], body["side"], body.get("response_ms"))
                self.send("{}", "application/json")
    return H


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", default="abc")
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    st = State(a.user)
    print(f"{len(st.pool)} pairs in pool, {st.prof.n_pairs} existing ratings. http://localhost:{a.port}")
    ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(st)).serve_forever()
