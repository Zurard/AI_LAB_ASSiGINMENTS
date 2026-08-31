# Simulated Annealing for the Traveling Salesman Problem
# Tour of important tourist locations in Rajasthan.
# Cost of an edge = great-circle distance in kilometres.

import math
import os
import random
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cities import CITIES, names


# ---------- distance model ----------

EARTH_RADIUS_KM = 6371.0


def haversine(lat1, lon1, lat2, lon2):
    """Great-circle distance between two GPS points, in kilometres."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def build_distance_matrix(city_list):
    n = len(city_list)
    dist = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = haversine(city_list[i][1], city_list[i][2],
                          city_list[j][1], city_list[j][2])
            dist[i][j] = d
            dist[j][i] = d
    return dist


# ---------- tour helpers ----------

def tour_cost(tour, dist):
    """Closed-cycle length: last city returns to the first."""
    total = 0.0
    n = len(tour)
    for i in range(n):
        total += dist[tour[i]][tour[(i + 1) % n]]
    return total


def rotate_to_start(tour, start=0):
    """Rotate a cycle so it is printed from a chosen home city (Jaipur)."""
    k = tour.index(start)
    return tour[k:] + tour[:k]


def two_opt_neighbor(tour):
    """Reverse a random segment of the tour (a 2-opt move)."""
    n = len(tour)
    i, j = sorted(random.sample(range(n), 2))
    # a 2-opt move needs at least two edges; skip no-op reversals
    if j - i < 1:
        return tour[:]
    neighbor = tour[:]
    neighbor[i:j + 1] = reversed(neighbor[i:j + 1])
    return neighbor


def nearest_neighbor_tour(dist, start=0):
    """Greedy constructive heuristic used only as a baseline."""
    n = len(dist)
    unvisited = set(range(n))
    unvisited.remove(start)
    tour = [start]
    current = start
    while unvisited:
        nxt = min(unvisited, key=lambda k: dist[current][k])
        tour.append(nxt)
        unvisited.remove(nxt)
        current = nxt
    return tour


def random_tour(n):
    tour = list(range(n))
    random.shuffle(tour)
    return tour


# ---------- simulated annealing ----------

def simulated_annealing(
    dist,
    t0=8000.0,
    t_min=1e-3,
    alpha=0.995,
    inner=80,
    seed=42,
    start_tour=None,
):
    """
    Minimise a TSP tour with geometric cooling and the Metropolis rule.

    t0     : starting temperature (high => more bad moves accepted)
    t_min  : stop when the temperature falls below this
    alpha  : cooling factor, T <- alpha * T   (0 < alpha < 1)
    inner  : candidate moves tried at each temperature
    """
    random.seed(seed)
    n = len(dist)
    current = start_tour[:] if start_tour is not None else random_tour(n)
    current_cost = tour_cost(current, dist)
    best = current[:]
    best_cost = current_cost

    history = [current_cost]
    temps = [t0]
    accepted = 0
    proposed = 0

    t = t0
    t_start = time.perf_counter()
    while t > t_min:
        for _ in range(inner):
            neighbor = two_opt_neighbor(current)
            neighbor_cost = tour_cost(neighbor, dist)
            delta = neighbor_cost - current_cost
            proposed += 1
            # always take an improvement; sometimes take a worse move
            if delta < 0 or random.random() < math.exp(-delta / t):
                current = neighbor
                current_cost = neighbor_cost
                accepted += 1
                if current_cost < best_cost:
                    best = current[:]
                    best_cost = current_cost
        history.append(best_cost)
        temps.append(t)
        t *= alpha

    elapsed = time.perf_counter() - t_start
    return {
        "tour": best,
        "cost": best_cost,
        "history": history,
        "temps": temps,
        "accepted": accepted,
        "proposed": proposed,
        "seconds": elapsed,
        "seed": seed,
        "t0": t0,
        "t_min": t_min,
        "alpha": alpha,
        "inner": inner,
    }


# ---------- plots ----------

def _xy(city_list, tour):
    xs = [city_list[i][2] for i in tour] + [city_list[tour[0]][2]]
    ys = [city_list[i][1] for i in tour] + [city_list[tour[0]][1]]
    return xs, ys


def plot_cities(city_list, path):
    fig, ax = plt.subplots(figsize=(9.5, 8.2))
    lons = [c[2] for c in city_list]
    lats = [c[1] for c in city_list]
    ax.scatter(lons, lats, s=46, c="#1f4e79", zorder=3)
    for name, lat, lon in city_list:
        ax.annotate(name, (lon, lat), textcoords="offset points",
                    xytext=(5, 4), fontsize=7.5)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Twenty-two tourist locations in Rajasthan")
    ax.grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_tour(city_list, tour, title, path, color="#c0392b"):
    fig, ax = plt.subplots(figsize=(9.5, 8.2))
    xs, ys = _xy(city_list, tour)
    ax.plot(xs, ys, "-", color=color, linewidth=1.6, alpha=0.9, zorder=2)
    ax.scatter(xs[:-1], ys[:-1], s=46, c="#1f4e79", zorder=3)
    # mark the home city
    home = tour[0]
    ax.scatter([city_list[home][2]], [city_list[home][1]],
               s=110, c="#f1c40f", edgecolors="black", zorder=4, label="Start / end")
    for idx in tour:
        name, lat, lon = city_list[idx]
        ax.annotate(name, (lon, lat), textcoords="offset points",
                    xytext=(5, 4), fontsize=7.5)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_cooling(history, temps, path):
    fig, ax1 = plt.subplots(figsize=(9.5, 4.8))
    steps = list(range(len(history)))
    ax1.plot(steps, history, color="#1f4e79", linewidth=1.6, label="Best tour cost")
    ax1.set_xlabel("Cooling step")
    ax1.set_ylabel("Best tour cost (km)", color="#1f4e79")
    ax1.tick_params(axis="y", labelcolor="#1f4e79")
    ax2 = ax1.twinx()
    ax2.plot(steps, temps, color="#c0392b", linewidth=1.2, alpha=0.75, label="Temperature")
    ax2.set_ylabel("Temperature", color="#c0392b")
    ax2.tick_params(axis="y", labelcolor="#c0392b")
    ax2.set_yscale("log")
    ax1.set_title("Simulated Annealing: cost and temperature while cooling")
    ax1.grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_compare(city_list, tours, titles, path):
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.4))
    colors = ["#7f8c8d", "#c0392b"]
    for ax, tour, title, color in zip(axes, tours, titles, colors):
        xs, ys = _xy(city_list, tour)
        ax.plot(xs, ys, "-", color=color, linewidth=1.5, alpha=0.9, zorder=2)
        ax.scatter(xs[:-1], ys[:-1], s=28, c="#1f4e79", zorder=3)
        home = tour[0]
        ax.scatter([city_list[home][2]], [city_list[home][1]],
                   s=80, c="#f1c40f", edgecolors="black", zorder=4)
        for idx in tour:
            ax.annotate(city_list[idx][0], (city_list[idx][2], city_list[idx][1]),
                        textcoords="offset points", xytext=(3, 3), fontsize=6)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(True, linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ---------- pretty printing ----------

def format_tour(tour, city_list):
    labels = [city_list[i][0] for i in tour]
    labels.append(city_list[tour[0]][0])
    return " -> ".join(labels)


def print_run(tag, result, city_list):
    shown = rotate_to_start(result["tour"], 0)
    print(tag)
    print("  cost     : %.2f km" % result["cost"])
    print("  time     : %.3f s" % result["seconds"])
    print("  accepted : %d / %d  (%.1f%%)" % (
        result["accepted"], result["proposed"],
        100.0 * result["accepted"] / max(1, result["proposed"])))
    print("  tour     :", format_tour(shown, city_list))
    print()


def run_experiment(fig_dir="figures"):
    os.makedirs(fig_dir, exist_ok=True)
    city_list = CITIES
    dist = build_distance_matrix(city_list)
    n = len(city_list)

    random.seed(1)
    init = random_tour(n)
    init_cost = tour_cost(init, dist)

    nn = nearest_neighbor_tour(dist, start=0)
    nn_cost = tour_cost(nn, dist)

    sa = simulated_annealing(dist, seed=42, start_tour=init)
    sa_from_nn = simulated_annealing(dist, seed=7, start_tour=nn[:], t0=2500.0)

    # a few extra seeds from random starts, to show SA is not one-shot
    extra = []
    for seed in (11, 23, 99):
        extra.append(simulated_annealing(dist, seed=seed))

    sa_tour = rotate_to_start(sa["tour"], 0)
    nn_tour = rotate_to_start(nn, 0)
    init_shown = rotate_to_start(init, 0)

    plot_cities(city_list, os.path.join(fig_dir, "cities_map.png"))
    plot_tour(city_list, init_shown,
              "Random initial tour  (%.1f km)" % init_cost,
              os.path.join(fig_dir, "initial_tour.png"),
              color="#7f8c8d")
    plot_tour(city_list, sa_tour,
              "Simulated Annealing tour  (%.1f km)" % sa["cost"],
              os.path.join(fig_dir, "sa_tour.png"),
              color="#c0392b")
    plot_tour(city_list, nn_tour,
              "Nearest-Neighbour tour  (%.1f km)" % nn_cost,
              os.path.join(fig_dir, "nn_tour.png"),
              color="#2980b9")
    plot_compare(
        city_list,
        [init_shown, sa_tour],
        ["Random start  (%.1f km)" % init_cost,
         "After Simulated Annealing  (%.1f km)" % sa["cost"]],
        os.path.join(fig_dir, "before_after.png"),
    )
    plot_cooling(sa["history"], sa["temps"], os.path.join(fig_dir, "cooling_curve.png"))

    return {
        "cities": city_list,
        "names": names(),
        "dist": dist,
        "init": init_shown,
        "init_cost": init_cost,
        "nn": nn_tour,
        "nn_cost": nn_cost,
        "sa": sa,
        "sa_tour": sa_tour,
        "sa_from_nn": sa_from_nn,
        "extra": extra,
        "fig_dir": fig_dir,
    }


def main():
    print("=" * 72)
    print("RAJASTHAN TSP  |  Simulated Annealing")
    print("=" * 72)
    print()
    print("Cities (%d):" % len(CITIES))
    for i, (name, lat, lon) in enumerate(CITIES):
        print("  %2d  %-14s  %7.4f N,  %7.4f E" % (i + 1, name, lat, lon))
    print()

    data = run_experiment()
    print("Random initial tour")
    print("  cost : %.2f km" % data["init_cost"])
    print("  tour :", format_tour(data["init"], CITIES))
    print()
    print("Nearest Neighbour (greedy baseline)")
    print("  cost : %.2f km" % data["nn_cost"])
    print("  tour :", format_tour(data["nn"], CITIES))
    print()
    print_run("Simulated Annealing  (random start, seed 42)", data["sa"], CITIES)
    print_run("Simulated Annealing  (NN start, seed 7)", data["sa_from_nn"], CITIES)
    for r in data["extra"]:
        print_run("Simulated Annealing  (seed %s)" % r["seed"], r, CITIES)

    print("Figures saved in", os.path.abspath(data["fig_dir"]))


if __name__ == "__main__":
    main()
