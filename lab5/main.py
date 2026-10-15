"""Print the Lab 5 GMM-EM and K-means results."""

import numpy as np

from experiment import NAMES, SPECS, run, save_figures, summarize

 
def main():
    data = save_figures(run())
    print("Lab 5  |  Gaussian Mixture EM and K-means")
    print("450 points, 3 clusters, 150 each")
    print()
    for spec, name in zip(SPECS, NAMES):
        print("  %-18s mean %s   std %s" % (name, spec["mean"], spec["std"]))
    print()
    gmm = data["gmm"]
    print("GMM-EM")
    print("  iterations:", len(gmm.loglik_))
    print("  log-likelihood: %.2f -> %.2f" % (gmm.loglik_[0], gmm.loglik_[-1]))
    print("  agreement with true labels: %.1f%%" % (100 * data["gmm_acc"]))
    print("  fitted std:")
    print(np.round(np.sqrt(data["variances_aligned"]), 3))
    print()
    print("K-means")
    print("  inertia: %.2f" % data["km"]["inertia"])
    print("  agreement with true labels: %.1f%%" % (100 * data["km_acc"]))
    print("  disagreements with GMM-EM: %d" % int(data["disagree"].sum()))
    for row in summarize(data):
        print("    %s: %d / %d" % (row["name"], row["disagree"], row["n"]))
    print()
    print("figures:", data["fig_dir"])


if __name__ == "__main__":
    main()
