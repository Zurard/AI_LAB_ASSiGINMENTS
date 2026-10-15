"""Gaussian mixture model with diagonal covariances, fit by EM."""

import numpy as np


def log_gaussian_diag(X, mean, var):
    """Log density of a diagonal Gaussian at each row of X."""
    var = np.maximum(var, 1e-8)
    diff = X - mean
    return -0.5 * (
        np.sum(np.log(2.0 * np.pi * var))
        + np.sum((diff * diff) / var, axis=1)
    )


class DiagonalGMM:
    def __init__(self, n_components=3, max_iter=100, tol=1e-5, seed=0, n_init=8):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed
        self.n_init = n_init

    def _one_run(self, X, rng):
        n, d = X.shape
        k = self.n_components
        means = X[rng.choice(n, k, replace=False)].copy()
        var = np.tile(np.var(X, axis=0), (k, 1))
        weights = np.full(k, 1.0 / k)
        history = []

        for _ in range(self.max_iter):
            # E-step
            log_joint = np.column_stack([
                np.log(weights[j] + 1e-300) + log_gaussian_diag(X, means[j], var[j])
                for j in range(k)
            ])
            m = log_joint.max(axis=1, keepdims=True)
            resp = np.exp(log_joint - m)
            resp /= resp.sum(axis=1, keepdims=True)
            ll = float(np.sum(m.ravel() + np.log(np.exp(log_joint - m).sum(axis=1))))
            history.append(ll)

            # M-step
            nk = resp.sum(axis=0)
            weights = nk / n
            means = (resp.T @ X) / np.maximum(nk[:, None], 1e-12)
            for j in range(k):
                diff = X - means[j]
                var[j] = (resp[:, j][:, None] * diff * diff).sum(axis=0) / max(nk[j], 1e-12)
            var = np.maximum(var, 1e-8)

            if len(history) > 1 and abs(history[-1] - history[-2]) < self.tol:
                break

        labels = resp.argmax(axis=1)
        return {
            "weights": weights,
            "means": means,
            "variances": var,
            "labels": labels,
            "responsibilities": resp,
            "loglik": np.array(history),
        }

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        rng = np.random.default_rng(self.seed)
        best = None
        for _ in range(self.n_init):
            run = self._one_run(X, rng)
            if best is None or run["loglik"][-1] > best["loglik"][-1]:
                best = run
        self.weights_ = best["weights"]
        self.means_ = best["means"]
        self.variances_ = best["variances"]
        self.labels_ = best["labels"]
        self.responsibilities_ = best["responsibilities"]
        self.loglik_ = best["loglik"]
        return self
