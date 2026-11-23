"""Blind hand-labelling page: the human labels images the VLM already labelled, WITHOUT
seeing the VLM's answers (showing them would anchor the human and inflate agreement).

  python src/label_app.py --n 100          # then open http://localhost:8766

Saves data/labels/human.jsonl. check_axes.py compares human vs VLM vs CLIP vs measured.
"""
from __future__ import annotations

import argparse
import html
import json
import os
import random
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

sys.path.insert(0, os.path.dirname(__file__))
from vlm_label import QUESTIONS, SCHEMA  # noqa: E402

VLM = "data/labels/vlm.jsonl"
HUMAN = "data/labels/human.jsonl"

PAGE = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Label Check</title><style>
:root{--bg:#f6f5f2;--fg:#1d1d1b;--mute:#6b6a66;--card:#fff;--line:#dedcd6;--acc:#2f5bd3;--on:#fff}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--fg:#ecebe7;--mute:#9b9a95;--card:#212120;--line:#353532;--acc:#7c9cff;--on:#111}}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.4 -apple-system,system-ui,sans-serif}
main{max-width:1100px;margin:0 auto;padding:16px;display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:20px}
@media (max-width:760px){main{grid-template-columns:1fr}} img{width:100%;border-radius:8px;position:sticky;top:12px}
.q{margin-bottom:10px} .q b{display:block;font-size:13px;margin-bottom:4px} .q .mute{font-weight:400}
.mute{color:var(--mute)} label{display:inline-block;margin:2px 4px 2px 0}
input{display:none} span.c{display:inline-block;padding:5px 10px;border:1px solid var(--line);border-radius:999px;background:var(--card);cursor:pointer;font-size:13px}
input:checked+span.c{background:var(--acc);color:var(--on);border-color:var(--acc)}
button{font:inherit;padding:10px 18px;border-radius:8px;border:0;background:var(--acc);color:var(--on);cursor:pointer;margin-top:8px}
header{padding:12px 16px;border-bottom:1px solid var(--line);display:flex;justify-content:space-between}
</style></head><body><header><b>Label check (blind)</b><span class="mute">__DONE__ / __TOTAL__ done</span></header>
<main><div><img src="/img/__UID__"></div><form method="post" action="/save"><input type="hidden" name="uid" value="__UID__">
__QUESTIONS__<button>Save &amp; next</button> <span class="mute">Pick "unsure" whenever it's genuinely ambiguous.</span></form></main></body></html>"""


def questions_html():
    out = []
    for k, vals in SCHEMA.items():
        chips = "".join(f'<label><input type="radio" name="{k}" value="{v}" required><span class="c">{v.replace("_", " ")}</span></label>'
                        for v in vals + ["unsure"])
        out.append(f'<div class="q"><b>{k} <span class="mute">· {html.escape(QUESTIONS[k])}</span></b>{chips}</div>')
    return "".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--port", type=int, default=8766)
    a = ap.parse_args()
    vlm = [json.loads(l)["uid"] for l in open(VLM)]
    random.Random(3).shuffle(vlm)
    queue = vlm[: a.n]
    qhtml = questions_html()

    def done():
        return {json.loads(l)["uid"] for l in open(HUMAN)} if os.path.exists(HUMAN) else set()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *x):
            pass

        def send(self, body, ctype="text/html; charset=utf-8", code=200, loc=None):
            b = body if isinstance(body, bytes) else body.encode()
            self.send_response(code)
            if loc:
                self.send_header("Location", loc)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            p = urlparse(self.path).path
            if p.startswith("/img/"):
                fp = f"data/images/{os.path.basename(p[5:])}.jpg"
                return self.send(open(fp, "rb").read(), "image/jpeg") if os.path.exists(fp) else self.send("", code=404)
            d = done()
            left = [u for u in queue if u not in d]
            if not left:
                return self.send("<p style='font:16px system-ui;padding:24px'>All done, thank you. You can close this tab.</p>")
            self.send(PAGE.replace("__UID__", left[0]).replace("__QUESTIONS__", qhtml)
                      .replace("__DONE__", str(len(queue) - len(left))).replace("__TOTAL__", str(len(queue))))

        def do_POST(self):
            from urllib.parse import parse_qs
            body = parse_qs(self.rfile.read(int(self.headers["Content-Length"])).decode())
            rec = {k: v[0] for k, v in body.items()}
            os.makedirs(os.path.dirname(HUMAN), exist_ok=True)
            with open(HUMAN, "a") as f:
                f.write(json.dumps(rec) + "\n")
            self.send("", code=303, loc="/")

    print(f"http://localhost:{a.port}  ({len(queue)} images, {len(done() & set(queue))} done)")
    ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()


if __name__ == "__main__":
    main()
