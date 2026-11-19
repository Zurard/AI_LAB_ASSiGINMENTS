# Lab 10 — Markov Decision Processes and Dynamic Programming

This lab has two parts:

1. **4×3 grid world** — solve the stochastic grid MDP with **value iteration** and
   study how the living reward changes the optimal policy.
2. **Gbike bicycle rental** — formulate the continuing finite MDP and solve the
   **original** and **modified** problems with **policy iteration**.

Everything is implemented from scratch with `numpy`, `scipy` and `matplotlib`.

---

## Files and how to run

```text
lab10/
├── main.py          # entry point: runs both parts
├── gridworld.py     # 4×3 grid MDP, value iteration, policy iteration check
├── gbike.py         # Gbike MDP, exact policy evaluation + policy iteration
├── validate.py      # correctness checks (transition sums, textbook values, Bellman residual)
├── README.md        # this write-up
└── outputs/         # generated figures, JSON and CSV files
```

```bash
cd lab10
python main.py                # both parts
python main.py --part grid    # only the grid world
python main.py --part gbike   # only Gbike
python validate.py            # run the correctness checks (about 5 seconds)
```

Requirements: Python 3.9+, `numpy`, `scipy`, `matplotlib`.

The run is deterministic; no random seed is needed. Grid-world results are
computed to a Bellman residual below `1e-10`, and the Gbike policy iteration is
verified independently against value iteration (maximum value difference
`8.6e-11`, maximum Bellman residual `8.0e-13`).

---

# Part 1 — 4×3 grid world

## 1.1 MDP formulation

The environment is the standard Russell & Norvig 4-column × 3-row grid:

```text
row 0:   (0,0)   (0,1)   (0,2)   (0,3) = +1
row 1:   (1,0)   WALL    (1,2)   (1,3) = -1
row 2:   (2,0)   (2,1)   (2,2)   (2,3)
          START
```

* **States:** the 11 non-wall cells; the two terminal cells are absorbing.
* **Actions:** `Up`, `Down`, `Left`, `Right`.
* **Transition model:** the intended action succeeds with probability `0.8`.
  With probability `0.1` each, the agent slips to one of the two directions at
  right angles to the intended direction. If a move (intended or slip) would
  leave the grid or enter the internal wall at `(1,1)`, the agent stays in its
  current cell.
* **Reward:** the living reward `r(s)` is received for every step out of a
  non-terminal state; the terminal values are `+1` and `-1`. This is the
  textbook convention

  \[
  V(s)=R(s)+\gamma \max_a \sum_{s'}P(s'\mid s,a)V(s'),
  \]

  with `V(+1)=+1`, `V(-1)=-1` fixed. It reproduces the textbook values exactly
  (`0.812, 0.868, 0.918, ...` for `r(s)=-0.04`).
* **Discount:** `γ = 1` for the canonical experiment, as in the textbook.
  For the reward sweep a discount `γ = 0.9` is used; see Section 1.4 for why a
  discount is required for positive living rewards.

### Transition example

From `(2,0)` taking `Up`:

* `0.8` → `(1,0)`
* `0.1` → `Left` is a wall, so stay at `(2,0)`
* `0.1` → `Right` gives `(2,1)`

From `(0,0)` taking `Up`, both the intended move and the `Left` slip hit the
boundary, so the probability of staying is `0.9` and the probability of moving
right is `0.1`.

## 1.2 Value iteration

The implementation repeatedly applies

\[
V_{k+1}(s)=R(s)+\gamma\max_a\sum_{s'}P(s'\mid s,a)V_k(s')
\]

for every non-terminal state, leaving terminal values fixed, until the largest
change is below `1e-10` (or a maximum number of sweeps is reached). The greedy
policy is then extracted from the converged value function.

Policy iteration is also implemented and used as an independent check: it solves
`(I - γP_π)V = R_π` exactly and improves the policy greedily. For the canonical
problem both methods agree to `7.4e-11` and produce the same policy.

## 1.3 Canonical result: `r(s) = -0.04`, `γ = 1`

Value iteration converges in **40 sweeps**. The values are exactly the standard
textbook values:

```text
 +0.812 > | +0.868 > | +0.918 > | +1.000 T
 +0.762 ^ |   WALL   | +0.660 ^ | -1.000 T
 +0.705 ^ | +0.655 < | +0.611 < | +0.388 <
```

The optimal policy sends the agent along the top row into the `+1` terminal
state, while avoiding the `-1` terminal state.

## 1.4 Reward sweep: `r(s) = -2, 0.1, 0.02, 1`

The assignment asks for the optimal value function for several living rewards.
Because no discount is specified for this part, the code reports both:

* the canonical `γ = 1` behaviour (including divergence), and
* a consistent `γ = 0.9` sweep, which gives finite value functions for every
  reward.

With `γ = 1`, a **positive** living reward makes the MDP *improper*: the agent
can wander forever in the non-terminal part of the grid and collect reward at
every step, so the optimal value is unbounded. The agent prefers an infinite
stream of small positive rewards to the finite `+1` terminal reward. This is not
a bug in value iteration — it is the correct behaviour of an undiscounted MDP
with positive rewards. The program detects this (the sweep does not converge)
and the divergence is plotted in `outputs/gridworld_gamma1_divergence.png`.

With `γ = 0.9`, all five rewards converge. Values below are rounded to three
decimals; arrows show the greedy policy, `T` marks a terminal cell.

### `r(s) = -0.04` — 37 sweeps

```text
 +0.509 > | +0.650 > | +0.795 > | +1.000 T
 +0.399 ^ |   WALL   | +0.486 ^ | -1.000 T
 +0.296 ^ | +0.254 > | +0.345 ^ | +0.130 <
```

### `r(s) = -2` — 36 sweeps

```text
 -6.107 > | -3.965 > | -1.738 > | +1.000 T
 -7.801 ^ |   WALL   | -3.354 > | -1.000 T
 -8.588 > | -7.101 > | -5.310 > | -3.514 ^
```

A large negative living reward makes every step expensive, so the agent takes
the shortest route to a terminal state. At `γ = 1` the same reward also
converges (35 sweeps, `V(start) = -10.815`).

### `r(s) = 0.1` — 130 sweeps

```text
 +1.000 ^ | +1.000 > | +1.000 > | +1.000 T
 +1.000 ^ |   WALL   | +1.000 < | -1.000 T
 +1.000 ^ | +1.000 < | +1.000 < | +1.000 v
```

The agent avoids both terminal states and circulates forever. The value of the
infinite loop is `0.1 / (1 - 0.9) = 1.0`, exactly the values shown.

### `r(s) = 0.02` — 121 sweeps

```text
 +0.713 > | +0.792 > | +0.874 > | +1.000 T
 +0.650 ^ |   WALL   | +0.615 ^ | -1.000 T
 +0.590 ^ | +0.543 < | +0.544 ^ | +0.363 v
```

The positive reward is small enough that reaching `+1` is still worthwhile in
most states, but the policy now hesitates near the `-1` terminal state.

### `r(s) = 1` — 263 sweeps

```text
 +10.000 ^ | +10.000 ^ | +10.000 < | +1.000 T
 +10.000 ^ |   WALL   | +10.000 < | -1.000 T
 +10.000 ^ | +10.000 ^ | +10.000 ^ | +10.000 v
```

The agent avoids the terminal states completely; the discounted value of
collecting `+1` forever is `1 / (1 - 0.9) = 10.0`.

### `γ = 1` behaviour after 300 sweeps

| `r(s)` | converged at 300 sweeps? | `V(start)` after 300 sweeps |
|:------:|:------------------------:|:---------------------------:|
| -0.04  | yes (40 sweeps)          | 0.7053                      |
| -2     | yes (35 sweeps)          | -10.8153                    |
| 0.1    | no — growing linearly    | 30.7641                     |
| 0.02   | no — growing linearly    | 6.9021                      |
| 1      | no — growing linearly    | 300.6112                    |

The full sweep figure is
[`outputs/gridworld_reward_sweep.png`](outputs/gridworld_reward_sweep.png) and
all values/policies are in
[`outputs/gridworld_results.json`](outputs/gridworld_results.json).

**Takeaway:** the living reward controls *how quickly* the agent wants to leave
the non-terminal part of the grid. Negative rewards encourage short episodes;
positive rewards encourage long episodes. With no discount and positive
rewards, “long” becomes infinite, so a discount is needed for a well-posed
finite-horizon value function.

---

# Part 2 — Gbike bicycle rental

## 2.1 Formulation of the continuing finite MDP

### State

\[
s=(c_1,c_2),\qquad 0\le c_1,c_2\le 20,
\]

where `c_i` is the number of bicycles at location `i` at the end of the day,
after the day's rentals and returns. There are `21 × 21 = 441` states.

### Action

\[
a\in\{-5,-4,\ldots,0,\ldots,4,5\}.
\]

A positive action moves `a` bicycles from location 1 to location 2; a negative
action moves `-a` bicycles in the opposite direction. An action is feasible only
if it respects the 20-bike capacity at both locations:

\[
0\le c_1-a\le 20,\qquad 0\le c_2+a\le 20,\qquad |a|\le 5.
\]

### Day-to-day dynamics

1. Bicycles are moved overnight: `n_1 = c_1 - a`, `n_2 = c_2 + a`.
2. Rental requests arrive at location `i`:
   `R_i ~ Poisson(λ_i)`, with `λ_1 = 3`, `λ_2 = 4`.
   The number served is `min(R_i, n_i)`, earning **INR 10** each.
3. Returns arrive at location `i`: `T_i ~ Poisson(μ_i)`, with `μ_1 = 3`,
   `μ_2 = 2`. They are added to the inventory, capped at 20.
4. The resulting inventory is the next state `s'`.

The request and return means are therefore:

| location | requests | returns |
|:--------:|:--------:|:-------:|
| 1        | 3        | 3       |
| 2        | 4        | 2       |

### Reward

For a move `a` and post-move inventories `n_1, n_2`:

\[
R(s,a)=10\left(\mathbb{E}[\min(R_1,n_1)]+\mathbb{E}[\min(R_2,n_2)]\right)
-\text{move}(a)-\text{park}(n_1,n_2).
\]

The expected number served at a location with `c` bicycles is

\[
\mathbb{E}[\min(R,c)]=\sum_{k=0}^{c-1}k\,\Pr(R=k)+c\,\Pr(R\ge c).
\]

**Original problem:** `move(a) = 2|a|`, no parking cost.

**Modified problem:**

\[
\text{move}(a)=
\begin{cases}
2|a|, & a\le 0,\\
0, & a=1,\\
2(a-1), & a>1,
\end{cases}
\]

because the employee shuttles the first bicycle from location 1 to location 2
for free. In addition,

\[
\text{park}(n_1,n_2)=4\cdot\mathbb{1}[n_1>10]+4\cdot\mathbb{1}[n_2>10],
\]

i.e. INR 4 for each location that keeps more than 10 bicycles overnight, after
any moving.

### Transition probabilities

Requests and returns are independent across the two locations, so the
transition distribution factorises:

\[
P(s'\mid s,a)=q^{(1)}_{n_1}(c_1')\,q^{(2)}_{n_2}(c_2'),
\]

where `q^(i)_n(c')` is the probability that location `i`, starting the day with
`n` bicycles, ends the day with `c'` bicycles after rentals and returns. Each
`q` is obtained by convolving the distribution of remaining bicycles after
rentals with the Poisson return distribution, then capping at 20. This
factorisation makes policy evaluation and improvement cheap.

### Objective

Maximise the discounted expected total reward:

\[
V^\pi(s)=\mathbb{E}_\pi\left[\sum_{t=0}^{\infty}\gamma^t R(s_t,a_t)\mid s_0=s\right],
\qquad \gamma=0.9.
\]

## 2.2 Policy iteration

Policy iteration alternates two steps until the policy is stable:

1. **Policy evaluation** — solve the linear system exactly:

   \[
   (I-\gamma P_\pi)V_\pi = R_\pi.
   \]

   Since there are only 441 states, a dense `441 × 441` solve is fast and gives
   the exact value function (no iterative evaluation error).

2. **Policy improvement** — for every state choose

   \[
   \pi'(s)=\arg\max_{a\ \text{feasible}}
   \left[R(s,a)+\gamma\sum_{s'}P(s'\mid s,a)V_\pi(s')\right].
   \]

   Ties keep the current action, which prevents cycling between equally good
   moves.

The implementation starts from the “move nothing” policy and reports the
number of states changed at each iteration.

## 2.3 Results

### Original problem

Policy iteration converges in **5 iterations**.

* `V(10,10) = 574.948 INR`
* `V(0,0) = 421.414 INR`
* `V(20,20) = 636.990 INR`

The optimal policy has the expected structure: when one location has many
bicycles and the other has few, move up to five bicycles toward the low-inventory
location. In the middle of the state space it is optimal to move nothing.

### Modified problem

Policy iteration converges in **4 iterations**.

* `V(10,10) = 580.963 INR`
* `V(0,0) = 429.946 INR`
* `V(20,20) = 603.491 INR`

The modified optimal policy is visibly different:

* The free 1 → 2 shuttle creates a large region where moving **one** bicycle
  from location 1 to location 2 is optimal even when both locations are below
  the parking threshold. This is free and shifts a bicycle toward the
  higher-demand location 2 (mean request rate 4 versus 3).
* The INR 4 second-lot charge creates threshold effects around 10 bicycles.
  For example, at `(13,11)` the optimal action is `-1`: moving one bicycle back
  to location 1 costs INR 2 but avoids the INR 4 second-lot charge at location 2,
  so it is worth it. Moving two or three bicycles toward location 2 can be
  optimal when it empties the first lot below 11 while filling the second.
* The value of `(20,20)` drops by about INR 33.5 relative to the original
  problem because both locations are over 10 bicycles and the manager cannot
  escape the two second-lot charges quickly.

### Comparison

| state   | original | modified | change |
|:-------:|---------:|---------:|-------:|
| (10,10) | 574.948  | 580.963  | +6.014 |
| (0,0)   | 421.414  | 429.946  | +8.532 |
| (20,0)  | 554.948  | 559.980  | +5.032 |
| (0,20)  | 567.769  | 563.861  | -3.908 |
| (20,20) | 636.990  | 603.491  | -33.498|
| (5,15)  | 577.226  | 573.861  | -3.365 |
| (15,5)  | 565.775  | 572.963  | +7.188 |
| (12,8)  | 571.686  | 578.963  | +7.276 |
| (8,12)  | 577.240  | 577.158  | -0.082 |

Across all 441 states, the optimal action changes in **242 states**, the mean
value change is **-1.358 INR**, the maximum gain is **+12.057 INR**, and the
minimum change is **-33.498 INR**. The free shuttle is a pure gain, while the
second-lot charge is a pure cost; the net effect depends on the state.

### Validation

The factorised transition model was checked against a brute-force enumeration of
requests and returns for a test state and action: the expected reward and
expected next-state value matched to machine precision. Policy iteration was
also checked against value iteration for both the original and modified MDPs:

* maximum `|V_PI - V_VI| = 8.6e-11`
* maximum Bellman residual `= 8.0e-13`

## 2.4 Note on `gbike.zip`

No `gbike.zip` file was present in the repository or in the local Downloads
folder, so there was no reference implementation to diff against. The
implementation above follows the standard Sutton & Barto formulation described
in the problem statement, and the original problem is solved alongside the
modified one so the two formulations can be compared directly.

---

## Outputs

Running `python main.py` writes:

| file | contents |
|------|----------|
| `outputs/gridworld_canonical.png` | canonical value function and policy (`r=-0.04`, `γ=1`) |
| `outputs/gridworld_reward_sweep.png` | value/policy panels for all five rewards (`γ=0.9`) |
| `outputs/gridworld_gamma1_divergence.png` | divergence of positive living rewards at `γ=1` |
| `outputs/gridworld_results.json` | all grid-world values, policies and iteration counts |
| `outputs/gbike_original.png` | original value function and optimal policy |
| `outputs/gbike_modified.png` | modified value function and optimal policy |
| `outputs/gbike_comparison.png` | original vs modified values and policies |
| `outputs/gbike_differences.png` | value difference and policy-change map |
| `outputs/gbike_original_policy.csv` | 21×21 original policy (integers, −5…5) |
| `outputs/gbike_modified_policy.csv` | 21×21 modified policy (integers, −5…5) |
| `outputs/gbike_original_values.csv` | 21×21 original state values |
| `outputs/gbike_modified_values.csv` | 21×21 modified state values |
| `outputs/gbike_results.json` | parameters, values, policies, convergence history |

---

## References

1. S. Russell and P. Norvig, *Artificial Intelligence: A Modern Approach*,
   3rd edition, Chapters 16, 17 and 21.
2. R. S. Sutton and A. G. Barto, *Reinforcement Learning: An Introduction*,
   2nd edition, Chapters 3 and 4 (the Jack's car rental problem).
3. R. S. Sutton and A. G. Barto, *Reinforcement Learning: An Introduction*,
   1st edition, Section 4.2 (policy iteration for Jack's car rental).
