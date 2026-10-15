"""Generate three Gaussian clusters and compare GMM-EM with K-means."""

import os
from itertools import permutations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np

from gmm import DiagonalGMM
from kmeans import kmeans


HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")

# Different means and different axis-aligned spreads. 150 points each.
SPECS = [
    {"mean": np.array([0.0, 0.0]), "std": np.array([0.55, 0.55])},
    {"mean": np.array([4.5, 0.4]), "std": np.array([2.30, 0.38])},
    {"mean": np.array([0.8, 4.8]), "std": np.array([0.38, 2.10])},
]
N_PER = 150
SEED = 7
COLORS = ["#1F3A5F", "#C0392B", "#1A7A6D"]
NAMES = ["compact", "wide horizontal", "tall vertical"]


def generate(seed=SEED):
    rng = np.random.default_rng(seed)
    parts, labels = [], []
    for k, spec in enumerate(SPECS):
        pts = rng.normal(loc=spec["mean"], scale=spec["std"], size=(N_PER, 2))
        parts.append(pts)
        labels.append(np.full(N_PER, k, dtype=int))
    X = np.vstack(parts)
    y = np.concatenate(labels)
    order = rng.permutation(len(X))
    return X[order], y[order]


def align_labels(pred, truth, k=3):
    """Permute predicted labels so they agree with truth as often as possible."""
    best_acc = -1.0
    best = pred
    for perm in permutations(range(k)):
        mapped = np.array([perm[p] for p in pred])
        acc = float((mapped == truth).mean())
        if acc > best_acc:
            best_acc = acc
            best = mapped
    return best, best_acc


def run():
    X, y = generate()
    gmm = DiagonalGMM(n_components=3, max_iter=80, tol=1e-5, seed=0, n_init=8).fit(X)
    km = kmeans(X, k=3, n_init=15, seed=1)
    gmm_labels, gmm_acc = align_labels(gmm.labels_, y)
    km_labels, km_acc = align_labels(km["labels"], y)
    disagree = gmm_labels != km_labels
    return {
        "X": X,
        "y": y,
        "gmm": gmm,
        "km": km,
        "gmm_labels": gmm_labels,
        "km_labels": km_labels,
        "gmm_acc": gmm_acc,
        "km_acc": km_acc,
        "disagree": disagree,
    }


def _style():
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 12,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def _scatter(ax, X, labels, title):
    for k in range(3):
        pts = X[labels == k]
        ax.scatter(pts[:, 0], pts[:, 1], s=16, c=COLORS[k], label=NAMES[k], alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="best")


def _ellipses(ax, means, variances, weights):
    for k in range(len(means)):
        for nsig, alpha in ((1, 0.9), (2, 0.55)):
            w = 2 * nsig * np.sqrt(variances[k, 0])
            h = 2 * nsig * np.sqrt(variances[k, 1])
            ell = Ellipse(
                xy=means[k], width=w, height=h,
                edgecolor=COLORS[k], facecolor="none",
                lw=1.6 if nsig == 2 else 1.0, alpha=alpha,
            )
            ax.add_patch(ell)
        ax.scatter(means[k, 0], means[k, 1], c=COLORS[k], marker="X", s=80, zorder=5)
    ax.legend(fontsize=8, loc="best")


def save_figures(data):
    os.makedirs(FIG, exist_ok=True)
    _style()
    X, y = data["X"], data["y"]
    gmm = data["gmm"]

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.6))
    _scatter(axes[0], X, y, "True clusters")
    axes[1].scatter(X[:, 0], X[:, 1], s=16, c="#5D6D7E", alpha=0.85)
    axes[1].set_title("Unlabeled data")
    axes[1].set_xlabel("x1")
    axes[1].set_ylabel("x2")
    axes[1].set_aspect("equal", adjustable="datalim")
    axes[1].grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "true_and_unlabeled.png"), dpi=140)
    plt.close(fig)

    # Map fitted components onto the true-label colours.
    # gmm_labels are already aligned, so component j that was mapped to true k
    # must be found from the original labels.
    # Rebuild means/variances in true-label order.
    order = []
    for k in range(3):
        members = gmm.labels_[data["gmm_labels"] == k]
        # most common original component among points now labelled k
        if len(members) == 0:
            order.append(k)
        else:
            order.append(int(np.bincount(members).argmax()))
    means = gmm.means_[order]
    variances = gmm.variances_[order]
    weights = gmm.weights_[order]

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    _scatter(ax, X, data["gmm_labels"], "GMM-EM assignments and fitted Gaussians")
    _ellipses(ax, means, variances, weights)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "gmm_fit.png"), dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    ax.plot(np.arange(1, len(gmm.loglik_) + 1), gmm.loglik_, color="#1F3A5F", marker="o", ms=3)
    ax.set_xlabel("EM iteration (best start)")
    ax.set_ylabel("log-likelihood")
    ax.set_title("GMM-EM convergence")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "loglik.png"), dpi=140)
    plt.close(fig)

    km_order = []
    for k in range(3):
        members = data["km"]["labels"][data["km_labels"] == k]
        km_order.append(int(np.bincount(members, minlength=3).argmax()))
    km_centers = data["km"]["centers"][km_order]

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    _scatter(ax, X, data["km_labels"], "K-means assignments")
    ax.scatter(km_centers[:, 0], km_centers[:, 1],
               c=COLORS, marker="X", s=90, zorder=5, edgecolors="black", linewidths=0.4)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "kmeans.png"), dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    same = ~data["disagree"]
    ax.scatter(X[same, 0], X[same, 1], s=14, c="#D5D8DC", label="same assignment")
    diff = X[data["disagree"]]
    diff_true = y[data["disagree"]]
    for k in range(3):
        pts = diff[diff_true == k]
        if len(pts):
            ax.scatter(pts[:, 0], pts[:, 1], s=22, c=COLORS[k], label="differ, true " + NAMES[k])
    ax.set_title("Points where K-means and GMM-EM disagree")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "disagree.png"), dpi=140)
    plt.close(fig)

    data["means_aligned"] = means
    data["variances_aligned"] = variances
    data["weights_aligned"] = weights
    data["km_centers"] = km_centers
    data["fig_dir"] = FIG
    return data


def summarize(data):
    y = data["y"]
    disagree = data["disagree"]
    by_cluster = []
    for k in range(3):
        mask = y == k
        by_cluster.append({
            "name": NAMES[k],
            "n": int(mask.sum()),
            "disagree": int((mask & disagree).sum()),
        })
    return by_cluster


if __name__ == "__main__":
    data = save_figures(run())
    print("gmm acc", round(data["gmm_acc"], 3), "iters", len(data["gmm"].loglik_))
    print("km acc", round(data["km_acc"], 3), "disagree", int(data["disagree"].sum()))
    print("ll", round(data["gmm"].loglik_[0], 2), "->", round(data["gmm"].loglik_[-1], 2))
    print(summarize(data))
    print("means", np.round(data["means_aligned"], 3))
    print("std", np.round(np.sqrt(data["variances_aligned"]), 3))
