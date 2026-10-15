"""Lloyd's K-means. Several random starts; the lowest inertia is kept."""

import numpy as np
 

def kmeans(X, k=3, max_iter=100, n_init=10, seed=1):
    X = np.asarray(X, dtype=float)
    n = len(X)
    rng = np.random.default_rng(seed)
    best = None

    for _ in range(n_init):
        centers = X[rng.choice(n, k, replace=False)].copy()
        labels = np.zeros(n, dtype=int)
        for _step in range(max_iter):
            dist = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            new_labels = dist.argmin(axis=1)
            new_centers = centers.copy()
            for j in range(k):
                members = X[new_labels == j]
                if len(members):
                    new_centers[j] = members.mean(axis=0)
            labels = new_labels
            if np.allclose(new_centers, centers):
                centers = new_centers
                break
            centers = new_centers
        inertia = float(((X - centers[labels]) ** 2).sum())
        if best is None or inertia < best["inertia"]:
            best = {"labels": labels.copy(), "centers": centers.copy(), "inertia": inertia}
    return best
