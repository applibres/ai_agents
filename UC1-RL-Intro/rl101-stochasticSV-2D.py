import numpy as np
import matplotlib.pyplot as plt

# Grid world setup: 3 rows x 4 cols
rows, cols = 3, 4
grid = np.zeros((rows, cols))

# Special states
invalid = (1, 1)   # wall / obstacle
trap = (1, 3)      # negative terminal
goal = (0, 3)      # positive terminal

# Rewards
living_reward = -2.0  # cost of each step into a non-terminal cell
R = np.full((rows, cols), living_reward)
R[trap] = -1
R[goal] = 1

# Mark invalid cell (nan to exclude)
grid[invalid] = np.nan
R[invalid] = np.nan

# Parameters
gamma = 0.9        # discount factor
theta = 0.0001     # convergence threshold
actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
action_names = {(-1, 0): "up", (1, 0): "down", (0, -1): "left", (0, 1): "right"}

# Stochastic transition model:
#   with probability p_intended the agent moves in the chosen direction,
#   with probability p_side it slips 90 degrees to either side.
p_intended = 0.8
p_side = 0.1  # per side (0.1 + 0.1 = 0.2 total slip)


def is_valid(state):
    r, c = state
    if r < 0 or r >= rows or c < 0 or c >= cols:
        return False
    if (r, c) == invalid:
        return False
    return True


def move(state, action):
    """Apply a single deterministic move, staying put if blocked."""
    r, c = state
    dr, dc = action
    new_state = (r + dr, c + dc)
    if not is_valid(new_state):
        new_state = state  # bump into wall / boundary -> stay
    return new_state


def perpendicular(action):
    """The two actions 90 degrees to the left and right of `action`."""
    dr, dc = action
    return [(-dc, dr), (dc, -dr)]


def get_transitions(state, action):
    """Stochastic transition: list of (probability, next_state)."""
    left, right = perpendicular(action)
    candidates = [
        (p_intended, action),
        (p_side, left),
        (p_side, right),
    ]
    # Collapse duplicate next_states (e.g. several outcomes bump the same wall).
    transitions = {}
    for prob, a in candidates:
        ns = move(state, a)
        transitions[ns] = transitions.get(ns, 0.0) + prob
    return [(prob, ns) for ns, prob in transitions.items()]


# Bellman update (stochastic: expectation over transition outcomes)
def bellman_update(V):
    new_V = np.copy(V)
    for r in range(rows):
        for c in range(cols):
            if (r, c) in [goal, trap, invalid]:
                continue  # skip terminal and invalid states
            q_values = []
            for a in actions:
                q = 0.0
                for prob, (nr, nc) in get_transitions((r, c), a):
                    q += prob * (R[nr, nc] + gamma * V[nr, nc])
                q_values.append(q)
            new_V[r, c] = max(q_values)
    return new_V


def extract_policy(V):
    """Greedy policy w.r.t. the converged values."""
    policy = {}
    for r in range(rows):
        for c in range(cols):
            if (r, c) in [goal, trap, invalid]:
                continue
            best_a, best_q = None, -np.inf
            for a in actions:
                q = 0.0
                for prob, (nr, nc) in get_transitions((r, c), a):
                    q += prob * (R[nr, nc] + gamma * V[nr, nc])
                if q > best_q:
                    best_q, best_a = q, a
            policy[(r, c)] = best_a
    return policy


# Value Iteration loop
V = np.zeros((rows, cols))
V[invalid] = np.nan
iteration = 0

while True:
    new_V = bellman_update(V)
    delta = np.nanmax(np.abs(new_V - V))
    V = new_V
    iteration += 1
    if delta < theta:
        break

print("Converged Values after", iteration, "iterations:")
print(np.round(V, 3))

policy = extract_policy(V)
print("\nGreedy policy:")
for r in range(rows):
    row_str = []
    for c in range(cols):
        if (r, c) == goal:
            row_str.append("GOAL")
        elif (r, c) == trap:
            row_str.append("TRAP")
        elif (r, c) == invalid:
            row_str.append("####")
        else:
            row_str.append(f"{action_names[policy[(r, c)]]:>5}")
    print(" ".join(f"{s:>5}" for s in row_str))

# Visualization: heatmap + greedy policy arrows
plt.figure(figsize=(6, 4))
plt.title("State Values (Stochastic Transitions, p={:.1f})".format(p_intended))
im = plt.imshow(V, cmap="coolwarm", interpolation="nearest")

arrow = {(-1, 0): (0, -0.3), (1, 0): (0, 0.3), (0, -1): (-0.3, 0), (0, 1): (0.3, 0)}
for r in range(rows):
    for c in range(cols):
        if np.isnan(V[r, c]):
            continue
        plt.text(c, r - 0.15, f"{V[r, c]:.2f}", ha='center', va='center', color='black')
        if (r, c) in policy:
            dx, dy = arrow[policy[(r, c)]]
            plt.arrow(c, r + 0.15, dx, dy, head_width=0.12, head_length=0.1,
                      fc='black', ec='black')

plt.colorbar(im, label="Value")
plt.show()
