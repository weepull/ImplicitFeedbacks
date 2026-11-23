"""Taste model: Bayesian Bradley-Terry over CLIP features with a population prior.

    P(user prefers A over B) = sigmoid( w_user . (x_A - x_B) )
    x = [CLIP image embedding (768), prompt alignment cos(img, caption), LAION aesthetic]
    prior:  w_user ~ N(w_global, (1/lam) I)        (w_global learned from other users)
    fit:    MAP by Newton; posterior covariance by Laplace approximation.

"P(user likes image)" is defined relative to a reference pool of typical generated
images: the probability the user would pick this image over a random pool image,
averaged over posterior samples of w_user. Reported with an 80% credible interval.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

EMB_DIM = 768
FEATS = EMB_DIM + 2  # + alignment + aesthetic


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


# ---------------------------------------------------------------- data access
class Store:
    """Image embeddings, caption embeddings, aesthetic scores, keyed by uid / caption."""

    def __init__(self, path: str = "data/embeddings.npz"):
        z = np.load(path, allow_pickle=False)
        self.uids = z["uids"]
        self.emb = z["emb"].astype(np.float32)
        self.aest = z["aesthetic"].astype(np.float32)
        self.idx = {u: i for i, u in enumerate(self.uids)}
        self.cap_idx = {c: i for i, c in enumerate(z["captions"])}
        self.cap_emb = z["cap_emb"].astype(np.float32)

    def features(self, uids, captions) -> np.ndarray:
        """Raw (unscaled) feature rows for images shown under the given captions."""
        ii = np.array([self.idx[u] for u in uids])
        e = self.emb[ii]
        if captions is None:
            align = np.zeros(len(ii), np.float32)
        else:
            t = self.cap_emb[np.array([self.cap_idx[c] for c in captions])]
            align = (e * t).sum(1)
        return np.hstack([e, align[:, None], self.aest[ii, None]])

    def pair_diffs(self, pairs: pd.DataFrame):
        """-> (X diffs (n, FEATS), y in {0,1}: 1 if image_0 won)."""
        xa = self.features(pairs.image_0_uid.values, pairs.caption.values)
        xb = self.features(pairs.image_1_uid.values, pairs.caption.values)
        y = (pairs.label_0.values > 0.5).astype(np.float64)
        return xa - xb, y


# ---------------------------------------------------------------- core fit
def fit_map(X, y, prior_mean, lam, iters: int = 50, tol: float = 1e-6, sample_weight=None, offset=None):
    """MAP logistic regression with Gaussian prior N(prior_mean, I/lam). Returns (w, posterior cov).

    sample_weight (0..1 per pair) is used by soft conditioning: a pair that only half
    matches the condition counts as half an observation.
    offset: fixed per-pair logit added to X @ w (used by the low-rank fit)."""
    d = X.shape[1]
    sw = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, float)
    off = np.zeros(len(y)) if offset is None else offset
    w = prior_mean.copy()
    for _ in range(iters):
        p = sigmoid(X @ w + off)
        g = X.T @ (sw * (p - y)) + lam * (w - prior_mean)
        H = (X * (sw * p * (1 - p))[:, None]).T @ X + lam * np.eye(d)
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < tol:
            break
    p = sigmoid(X @ w + off)
    H = (X * (sw * p * (1 - p))[:, None]).T @ X + lam * np.eye(d)
    return w, np.linalg.inv(H)


# ---------------------------------------------------------------- low-rank personalisation
def make_basis(Xs_global: np.ndarray, k: int) -> np.ndarray:
    """(FEATS, k+2) orthonormal basis for personal deviations: the top-k principal directions
    of the embedding part of population pair differences (where compared images actually
    differ), plus the prompt-alignment and aesthetic weights as their own directions."""
    E = Xs_global[:, :EMB_DIM]
    _, _, Vt = np.linalg.svd(E - E.mean(0), full_matrices=False)
    P = np.zeros((FEATS, k + 2))
    P[:EMB_DIM, :k] = Vt[:k].T
    P[EMB_DIM, k] = 1.0
    P[EMB_DIM + 1, k + 1] = 1.0
    return P


def fit_lowrank(user, Xs, y, prior_mean, P, lam, sample_weight=None):
    """w_user = prior_mean + P a,  a ~ N(0, I/lam). Only k+2 numbers are learned per user."""
    if len(y) == 0:
        return TasteProfile(user, prior_mean.copy(), P @ P.T / lam, 0)
    Z = Xs @ P
    a, cov_a = fit_map(Z, y, np.zeros(P.shape[1]), lam, sample_weight=sample_weight, offset=Xs @ prior_mean)
    return TasteProfile(user, prior_mean + P @ a, P @ cov_a @ P.T, len(y))


@dataclass
class Population:
    """Feature scaling + population-level weights, learned from non-target users."""
    scale: np.ndarray  # per-feature std of pair diffs; features are divided by it
    w: np.ndarray
    lam: float

    @classmethod
    def fit(cls, X, y, lams=(1, 3, 10, 30, 100, 300), seed=0):
        scale = X.std(0) + 1e-8
        Xs = X / scale
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(y))
        cut = int(0.8 * len(y))
        tr, va = perm[:cut], perm[cut:]
        best = None
        for lam in lams:
            w, _ = fit_map(Xs[tr], y[tr], np.zeros(FEATS), lam)
            ll = log_loss(y[va], sigmoid(Xs[va] @ w))
            if best is None or ll < best[0]:
                best = (ll, lam)
        w, _ = fit_map(Xs, y, np.zeros(FEATS), best[1])
        return cls(scale=scale, w=w, lam=best[1])


@dataclass
class TasteProfile:
    """A user's posterior over taste weights (in scaled feature space)."""
    user: str
    mean: np.ndarray
    cov: np.ndarray
    n_pairs: int

    @classmethod
    def fit(cls, user, Xs, y, prior_mean, lam, sample_weight=None):
        if len(y) == 0:
            return cls(user, prior_mean.copy(), np.eye(FEATS) / lam, 0)
        w, cov = fit_map(Xs, y, prior_mean, lam, sample_weight=sample_weight)
        return cls(user, w, cov, len(y))

    # ---- predictions
    def p_prefer(self, Xs_diff):
        """P(first > second), integrating over the posterior (probit approximation)."""
        mu = Xs_diff @ self.mean
        var = np.einsum("ij,jk,ik->i", Xs_diff, self.cov, Xs_diff)
        return sigmoid(mu / np.sqrt(1 + np.pi * var / 8))

    def score(self, Xs):
        return Xs @ self.mean

    def samples(self, n=200, seed=0):
        rng = np.random.default_rng(seed)
        # eigen-decomposition instead of Cholesky: low-rank covariances are only semi-definite
        vals, vecs = np.linalg.eigh((self.cov + self.cov.T) / 2)
        L = vecs * np.sqrt(np.clip(vals, 0, None))
        return self.mean + rng.standard_normal((n, len(self.mean))) @ L.T

    def p_like(self, Xs, Xs_ref, n_samples=200, ci=(10, 90)):
        """P(user picks image over a random reference image), mean + credible interval.

        Uses taste features only when Xs has no caption context (alignment column = 0)."""
        W = self.samples(n_samples)                       # (S, F)
        s = Xs @ W.T                                      # (n, S)
        r = Xs_ref @ W.T                                  # (m, S)
        # mean over reference images of sigmoid(s - r): (n, S)
        p = sigmoid(s[:, None, :] - r[None, :, :]).mean(1)
        return p.mean(1), np.percentile(p, ci[0], 1), np.percentile(p, ci[1], 1)

    def save(self, path):
        np.savez(path, user=self.user, mean=self.mean, cov=self.cov, n_pairs=self.n_pairs)

    @classmethod
    def load(cls, path):
        z = np.load(path)
        return cls(str(z["user"]), z["mean"], z["cov"], int(z["n_pairs"]))


# ---------------------------------------------------------------- metrics
def log_loss(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def accuracy(y, p):
    hit = ((p > 0.5) == (y > 0.5)).astype(float)
    hit[p == 0.5] = 0.5  # no prediction = coin flip, not "always image_1"
    return float(hit.mean())


def ece(y, p, bins=10):
    """Expected calibration error on the symmetric form: fold p<0.5 onto p>0.5."""
    conf = np.where(p >= 0.5, p, 1 - p)
    hit = np.where(p >= 0.5, y, 1 - y)
    edges = np.linspace(0.5, 1.0, bins // 2 + 1)
    tot = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & (conf < hi if hi < 1 else conf <= hi)
        if m.any():
            tot += m.mean() * abs(conf[m].mean() - hit[m].mean())
    return float(tot)
