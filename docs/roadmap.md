# Roadmap and parked decisions

## Parked: neural conditional taste model ("option D")

**What:** a network that scores (image, context, user) → preference, learning on its
own how preference depends on context (e.g. "close-ups in portraits, wide shots in
landscapes"), instead of conditioning on a phrase someone types (soft conditioning,
which is what is built now).

**Why parked:** with ~465 pairs per stand-in user it would overfit and be hard to
explain. Soft conditioning works with the data we have and gives honest uncertainty.

**Go / no-go test:** if soft-conditioned profiles predict held-out choices better than
unconditioned ones, context-dependent taste is real in this data → build D.
If conditioning does not help, D will not find signal either.

**Data, when ready:** the full Pick-a-Pic v1 train split, 354k decisive pairs from
4,108 users (68 users with >1,000 pairs). We currently use 80 of 387 shards.
- Remaining 307 shards ≈ 154 GB ≈ 3.5 h at the ~13 MB/s measured on this machine.
- Keep embeddings only, not JPEGs (≈ 1 GB on disk instead of ≈ 90 GB).
- `python src/build_dataset.py --shards 387 ...` resumes from where the 80-shard run stopped.
  The image-saving step in `build_dataset.py` would need a flag to skip JPEGs.

**Shape of the model (sketch):** shared MLP over [image embedding, context embedding]
plus a small per-user embedding; trained across all users; new users (e.g. `abc`) fit only
their user embedding from their ratings, with the population as prior.
