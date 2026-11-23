"""Stream Pick-a-Pic v1 train shards -> CLIP embeddings for the pairs we need.

Image URLs in the dataset are dead, and shards have 1000-row row groups, so the
only way to get pixels is to download whole shards (~500MB each). We stream:
download shard -> embed the images we need -> save JPEGs we want to look at ->
delete shard. Resumable: shards with an existing data/emb/<idx>.npz are skipped.

Selection:
  * target users  = top-K users by decisive pair count (stand-ins for the target user).
                    All their pairs in every processed shard are kept.
  * global pairs  = every decisive pair from other users in the first G shards.
                    Used to train the population prior, and as the target user's rating pool.

Usage:
  python src/build_dataset.py --users 12 --shards 387 --global-shards 12
  python src/build_dataset.py --merge        # combine per-shard npz into data/embeddings.npz
"""
from __future__ import annotations

import argparse
import glob
import os
import queue
import threading

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import subprocess

REPO = "pickapic-anonymous/pickapic_v1"
RAW = "data/raw"
EMB = "data/emb"
IMG = "data/images"


def decisive(meta: pd.DataFrame) -> pd.DataFrame:
    return meta[meta.has_label & meta.are_different & (meta.label_0 != 0.5)]


def select(meta: pd.DataFrame, k_users: int, global_shards: list[str]) -> pd.DataFrame:
    d = decisive(meta).copy()
    targets = d.user_id.value_counts().index[:k_users]
    d["role"] = np.where(d.user_id.isin(targets), "target", "")
    d.loc[(d.role == "") & d.shard.isin(global_shards), "role"] = "global"
    return d[d.role != ""]


def downloader(shards: list[str], q: queue.Queue):
    for s in shards:
        path = f"{RAW}/{os.path.basename(s)}"
        url = f"https://huggingface.co/datasets/{REPO}/resolve/main/{s}"
        for attempt in range(5):  # curl is far more reliable here than hf_hub_download
            r = subprocess.run(["curl", "-sfL", "--retry", "3", "-o", path, url])
            if r.returncode == 0:
                break
        else:
            print(f"FAILED to download {s}", flush=True)
            continue
        q.put((s, path))
    q.put(None)


def run(args):
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from clip_model import Clip

    meta = pd.read_parquet("data/meta.parquet")
    shards = open("data/train_shards.txt").read().split()[: args.shards]
    sel = select(meta, args.users, shards[: args.global_shards])
    sel.to_parquet("data/pairs.parquet")
    print(sel.role.value_counts().to_dict(), "target users:", sel[sel.role == "target"].user_id.nunique())

    os.makedirs(EMB, exist_ok=True)
    os.makedirs(IMG, exist_ok=True)
    todo = [s for s in shards if not os.path.exists(f"{EMB}/{shards.index(s):03d}.npz")]
    print(f"{len(todo)} shards to process")

    clip = Clip()
    q: queue.Queue = queue.Queue(maxsize=args.prefetch)
    # several download threads, each owning a slice, all feeding one queue
    slices = [todo[i :: args.dl_threads] for i in range(args.dl_threads)]
    for sl in slices:
        threading.Thread(target=downloader, args=(sl, q), daemon=True).start()

    done_threads = 0
    n = 0
    while done_threads < args.dl_threads:
        item = q.get()
        if item is None:
            done_threads += 1
            continue
        shard, path = item
        idx = shards.index(shard)
        rows = sel[sel.shard == shard]
        t = pq.read_table(path, columns=["image_0_uid", "image_1_uid", "jpg_0", "jpg_1"]).to_pandas()
        t = t.iloc[rows.row.values]
        uids, blobs = [], []
        for side in (0, 1):
            uids += list(t[f"image_{side}_uid"])
            blobs += list(t[f"jpg_{side}"])
        uniq = dict(zip(uids, blobs))
        uids, blobs = list(uniq), list(uniq.values())
        emb = clip.embed_images(blobs) if blobs else np.zeros((0, 768), np.float32)
        for u, b in zip(uids, blobs):
            fp = f"{IMG}/{u}.jpg"
            if not os.path.exists(fp):
                with open(fp, "wb") as f:
                    f.write(b)
        np.savez(f"{EMB}/{idx:03d}.npz", uids=np.array(uids), emb=emb.astype(np.float16))
        os.remove(path)
        n += 1
        print(f"[{n}/{len(todo)}] shard {idx}: {len(rows)} pairs, {len(uids)} images", flush=True)


def merge():
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from clip_model import Aesthetic, Clip

    uids, embs = [], []
    for f in sorted(glob.glob(f"{EMB}/*.npz")):
        z = np.load(f)
        uids += list(z["uids"])
        embs.append(z["emb"].astype(np.float32))
    emb = np.concatenate(embs)
    uids_arr = np.array(uids)
    _, keep = np.unique(uids_arr, return_index=True)
    uids_arr, emb = uids_arr[keep], emb[keep]
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)

    pairs = pd.read_parquet("data/pairs.parquet")
    have = set(uids_arr)
    pairs = pairs[pairs.image_0_uid.isin(have) & pairs.image_1_uid.isin(have)]
    pairs.to_parquet("data/pairs_ready.parquet")  # selection in pairs.parquet stays intact

    captions = sorted(pairs.caption.unique())
    clip = Clip()
    cap_emb = clip.embed_texts(captions)
    aest = Aesthetic().score(emb)
    np.savez("data/embeddings.npz", uids=uids_arr, emb=emb, aesthetic=aest,
             captions=np.array(captions), cap_emb=cap_emb)
    print(f"{len(uids_arr)} images, {len(captions)} captions, {len(pairs)} pairs",
          pairs.role.value_counts().to_dict())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=12)
    ap.add_argument("--shards", type=int, default=387)
    ap.add_argument("--global-shards", type=int, default=12)
    ap.add_argument("--dl-threads", type=int, default=4)
    ap.add_argument("--prefetch", type=int, default=4)
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    merge() if a.merge else run(a)
