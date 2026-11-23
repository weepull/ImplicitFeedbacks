# Day 1: What Can a Model See in an Image?

**Goal:** not a reranker, not a feedback loop. One question answered
empirically — given an AI-generated image, what signal can actually be
extracted from it, and is that signal good enough to tell an action scene from
a calm one?

By the end of the day you should be looking at a table of numbers per image and
able to say which columns are real and which are noise.

---

## Why this first

Everything downstream depends on the image representation. If CLIP cannot tell
an action scene from an establishing shot, no amount of preference learning
fixes it. Find that out today, on twenty images, not in three weeks on a
trained model.

---

## What is extractable, three tiers

### Tier 1: the embedding

One vector per image, 512-d from CLIP ViT-B/32.

Captures: subject matter, scene type, coarse composition, rough style, mood.
Misses: fine detail, text in image, counting, precise spatial relations, and
technical quality. A beautifully composed, badly blurred image embeds as a
beautiful composition.

Key property: image and text land in the **same space**, so you can score an
image against a text phrase directly.

### Tier 2: text probing (the useful one)

Because of that shared space, compare an image against a list of contrasting
phrase pairs and read off which side wins:

```
"motion blur, dynamic action"   vs  "sharp, static, still"
"low angle hero shot"           vs  "eye level neutral shot"
"wide establishing shot"        vs  "tight close-up"
"dramatic high-contrast light"  vs  "flat even lighting"
"crowded busy frame"            vs  "minimal empty frame"
```

Output: interpretable, named axes. Same model, no training, no extra cost.

This is what later answers "why did this image fail for this brief" instead of
just "it scored low."

### Tier 3: cheap classical metrics

- Sharpness: variance of the Laplacian
- Brightness, contrast: grayscale mean and std
- Saturation: mean of the HSV S channel
- Dominant colours: k-means on downsampled pixels
- Person/face count and relative size

These catch the technical failures CLIP is blind to. Pennies to compute.

### Also worth knowing

A **LAION aesthetic scorer** exists: a small pre-trained MLP that runs on CLIP
embeddings and predicts human aesthetic ratings. It is a drop-in extra column
and a useful contrast to brief-adherence, because it measures "pretty", which
is exactly the thing reviewers confuse with "correct".

---

## Environment

Mac, CPU is fine. ViT-B/32 is about 150MB and embeds an image in well under a
second.

```
pip install open_clip_torch pillow numpy opencv-python pandas
```

Model: `ViT-B-32`, pretrained `laion2b_s34b_b79k`.

Alternative if the API annoys you: `sentence-transformers` wraps CLIP with a
simpler interface. Either is fine; do not spend the morning choosing.

---

## Build steps

### Step 1: image set

Collect 20 to 30 generated images spanning at least three clearly different
intents (action scene, calm establishing shot, close-up portrait). Label them
by hand in a CSV. This hand labelling is your ground truth for the whole day;
do not skip it or guess it later.

### Step 2: embedding extraction

```python
def embed_image(path) -> np.ndarray:    # (512,) L2-normalised
    # TODO: load, preprocess, encode, normalise
    ...

def embed_text(phrases: list[str]) -> np.ndarray:   # (n, 512) normalised
    # TODO
    ...
```

**Verify before going further.** Cosine similarity between two action images
should exceed similarity between an action image and a portrait. If it doesn't,
something is wrong with preprocessing or normalisation. Check this with three
images and a print statement, not with a model.

### Step 3: the probe extractor

```python
PROBES = {
    "motion":      ("motion blur, dynamic action", "sharp, static, still"),
    "angle":       ("low angle hero shot", "eye level neutral shot"),
    "framing":     ("wide establishing shot", "tight close-up"),
    "lighting":    ("dramatic high-contrast lighting", "flat even lighting"),
    "density":     ("crowded busy frame", "minimal empty frame"),
    # TODO: add 3-5 more that matter for your actual briefs
}

def probe(img_emb, probes) -> dict[str, float]:
    # For each axis: score = sim(img, positive) - sim(img, negative)
    # Positive means the first phrase wins. Range roughly [-0.2, 0.2].
    # TODO
    ...
```

Design note: use **contrasting pairs**, not single phrases. A raw similarity to
"motion blur" is uncalibrated and tells you little. The difference between two
opposing phrases is far more stable.

### Step 4: classical metrics

```python
def cheap_metrics(path) -> dict[str, float]:
    # TODO: sharpness (Laplacian var), brightness, contrast, saturation
    ...
```

### Step 5: the table

Run everything over the image set, write one row per image: filename, hand
label, all probe scores, all classical metrics.

**Then actually look at it.** Open it, read it, sort by columns. This step is
the point of the day and it is the one people skip.

---

## Sanity checks that tell you if it worked

1. **Separation.** Group by your hand label and compare mean probe scores. Do
   action images score higher on `motion` than portraits do? If the separation
   isn't visible by eye in a group-by, it isn't there.

2. **Intent similarity.** Embed the text "an action scene" and score every image
   against it. Do your hand-labelled action images rank top? This is the
   zero-training baseline for the whole project, so measure it today.

3. **Where CLIP is blind.** Find an image you consider technically poor (blurry,
   bad artifacts) and check whether any CLIP-derived column noticed. Probably
   not. Confirm the classical metrics caught it. This tells you which tier each
   kind of failure lives in.

4. **Probe stability.** Reword one probe pair slightly and see how much the
   scores move. Large movement means the probe is unreliable and needs better
   phrasing or should be dropped.

---

## What you'll know by the end

- Whether CLIP separates your actual intents, measured rather than assumed
- Which probe axes are informative and which are noise
- What CLIP cannot see, and which cheap metric covers each gap
- A zero-training baseline number for intent matching

That last one matters more than it sounds. Any learned reranker you build later
has to beat it, and having the number from day one means you find out
immediately rather than after weeks of work.

---

## Deliberately out of scope today

No reranker. No training. No feedback collection. No UI. No database.

If you find yourself writing a training loop, you've drifted.

---

## If it goes badly

The most likely failure is that CLIP separates subject matter well but intent
poorly, because "action scene" is about composition and motion rather than
content. If so, the probe axes become the primary representation rather than a
supplement, and the next step is widening and validating the probe set rather
than reaching for a bigger model.

Second most likely: your generated images are too stylistically uniform for
embeddings to separate them, because they all came from one model with one
style. If so, the probe axes are again the answer, since they measure
composition rather than style.

Both outcomes are useful. Neither means stop.
