"""Scan metadata columns (no image bytes) of all Pick-a-Pic v1 train shards via HTTP range reads."""
import sys, pandas as pd, pyarrow.parquet as pq
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import HfFileSystem

COLS = ["ranking_id", "user_id", "caption", "created_at", "has_label", "are_different",
        "label_0", "label_1", "image_0_uid", "image_1_uid", "image_0_url", "image_1_url",
        "model_0", "model_1"]
fs = HfFileSystem()
shards = open("data/train_shards.txt").read().split()

def read(path):
    with fs.open(f"datasets/pickapic-anonymous/pickapic_v1/{path}", "rb", block_size=2**20) as f:
        pf = pq.ParquetFile(f)
        df = pf.read(columns=COLS).to_pandas()
    df["shard"] = path
    df["row"] = range(len(df))
    return df

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else len(shards)
    with ThreadPoolExecutor(16) as ex:
        dfs = list(ex.map(read, shards[:n]))
    meta = pd.concat(dfs, ignore_index=True)
    meta.to_parquet("data/meta.parquet")
    print(len(meta))
