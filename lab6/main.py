# Lab 6 -- i think we have a scrammbled image and we need to reconstruct using some adversial search 

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import random
import math

# Load the supplied 512x512 scrambled grayscale image
img = sio.loadmat("scrambled_lena.mat")["Iscrambled"]

# Treat the image as a 4x4 jigsaw (16 tiles, each 128x128)
GRID = 4
SIZE = img.shape[0] // GRID
tiles = [
    img[i*SIZE:(i+1)*SIZE, j*SIZE:(j+1)*SIZE].copy()
    for i in range(GRID) for j in range(GRID)
]
N = len(tiles)

def edge_cost(a, b, side):
    """Low cost means the touching edges of tiles look compatible."""
    a = a.astype(np.float32)
    b = b.astype(np.float32)

    if side == "right":
        edge_a, edge_b = a[:, -1], b[:, 0]
        grad_a = a[:, -1] - a[:, -2]
        grad_b = b[:, 1] - b[:, 0]
    else:  # a is above b
        edge_a, edge_b = a[-1, :], b[0, :]
        grad_a = a[-1, :] - a[-2, :]
        grad_b = b[1, :] - b[0, :]

    pixel_cost = np.mean((edge_a - edge_b) ** 2)
    gradient_cost = np.mean((grad_a - grad_b) ** 2)
    return float(pixel_cost + 0.5 * gradient_cost)

# Precompute compatibility costs for every ordered pair of tiles.
# This makes each simulated-annealing iteration fast.
right_cost = np.array([
    [edge_cost(tiles[a], tiles[b], "right") for b in range(N)]
    for a in range(N)
])
bottom_cost = np.array([
    [edge_cost(tiles[a], tiles[b], "bottom") for b in range(N)]
    for a in range(N)
])

def energy(board):
    """MRF energy: sum of costs for all horizontal and vertical neighbors."""
    total = 0.0
    for r in range(GRID):
        for c in range(GRID):
            tile_id = board[r, c]
            if c < GRID - 1:
                total += right_cost[tile_id, board[r, c+1]]
            if r < GRID - 1:
                total += bottom_cost[tile_id, board[r+1, c]]
    return total

# Simulated annealing with multiple random restarts.
# The best board across all restarts is retained.
rng = np.random.default_rng(42)
py_rng = random.Random(42)
best_board = None
best_energy = float("inf")

RESTARTS = 50
STEPS = 30000
INITIAL_TEMPERATURE = 10000.0
COOLING_RATE = 0.9998

for restart in range(RESTARTS):
    board = rng.permutation(N).reshape(GRID, GRID)
    current_energy = energy(board)
    temperature = INITIAL_TEMPERATURE

    for step in range(STEPS):
        # Propose swapping two randomly selected tile positions.
        a, b = py_rng.sample(range(N), 2)
        r1, c1 = divmod(a, GRID)
        r2, c2 = divmod(b, GRID)
        va, vb = board[r1, c1], board[r2, c2]
        board[r1, c1], board[r2, c2] = vb, va

        new_energy = energy(board)
        difference = new_energy - current_energy

        # Always accept an improvement; sometimes accept a worse move
        # to escape local minima. This probability falls as temperature cools.
        if difference < 0 or py_rng.random() < math.exp(
            -difference / max(temperature, 0.001)
        ):
            current_energy = new_energy
            if current_energy < best_energy:
                best_board = board.copy()
                best_energy = current_energy
        else:
            # Undo rejected swap.
            board[r1, c1], board[r2, c2] = va, vb

        temperature *= COOLING_RATE

    print(f"Restart {restart + 1}/{RESTARTS} complete; best energy = {best_energy:.3f}")

# Reconstruct the image from the best tile arrangement.
result = np.zeros_like(img)
for r in range(GRID):
    for c in range(GRID):
        result[r*SIZE:(r+1)*SIZE, c*SIZE:(c+1)*SIZE] = tiles[best_board[r, c]]

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img, cmap="gray")
plt.title("Scrambled Image")
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(result, cmap="gray")
plt.title("Reconstructed Lena")
plt.axis("off")
plt.tight_layout()
plt.savefig("lena_reconstructed.png", dpi=200)
plt.show()

sio.savemat("lena_result.mat", {"result": result})
print("Best MRF energy:", best_energy)
print("Tile arrangement (tile IDs, zero-based):")
print(best_board)
print("Saved lena_reconstructed.png and lena_result.mat")
