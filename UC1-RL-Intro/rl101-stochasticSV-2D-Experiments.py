import numpy as np
import matplotlib.pyplot as plt

# Grid world setup: 3 rows x 4 cols
rows, cols = 3, 4

# Special states
invalid = (1, 1)   # wall / obstacle
trap = (1, 3)      # negative terminal
goal = (0, 3)      # positive terminal

theta = 0.0001     # convergence threshold
actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
action_names = {(-1, 0): "up", (1, 0): "down", (0, -1): "left", (0, 1): "right"}
arrow = {(-1, 0): (0, -0.3), (1, 0): (0, 0.3), (0, -1): (-0.3, 0), (0, 1): (0.3, 0)}


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


def get_transitions(state, action, p_intended, p_side):
    """Stochastic transition: list of (probability, next_state)."""
    left, right = perpendicular(action)
    candidates = [
        (p_intended, action),
        (p_side, left),
        (p_side, right),
    ]
    transitions = {}
    for prob, a in candidates:
        ns = move(state, a)
        transitions[ns] = transitions.get(ns, 0.0) + prob
    return [(prob, ns) for ns, prob in transitions.items()]


def bellman_update(V, R, gamma, p_intended, p_side):
    new_V = np.copy(V)
    for r in range(rows):
        for c in range(cols):
            if (r, c) in [goal, trap, invalid]:
                continue
            q_values = []
            for a in actions:
                q = 0.0
                for prob, (nr, nc) in get_transitions((r, c), a, p_intended, p_side):
                    q += prob * (R[nr, nc] + gamma * V[nr, nc])
                q_values.append(q)
            new_V[r, c] = max(q_values)
    return new_V


def extract_policy(V, R, gamma, p_intended, p_side):
    policy = {}
    for r in range(rows):
        for c in range(cols):
            if (r, c) in [goal, trap, invalid]:
                continue
            best_a, best_q = None, -np.inf
            for a in actions:
                q = 0.0
                for prob, (nr, nc) in get_transitions((r, c), a, p_intended, p_side):
                    q += prob * (R[nr, nc] + gamma * V[nr, nc])
                if q > best_q:
                    best_q, best_a = q, a
            policy[(r, c)] = best_a
    return policy


def run_experiment(gamma, p_intended, p_side, living_reward, label="", plot=True):
    """Run value iteration with the given parameters and return (V, policy)."""
    R = np.full((rows, cols), living_reward)
    R[trap] = -1
    R[goal] = 1
    R[invalid] = np.nan

    V = np.zeros((rows, cols))
    V[invalid] = np.nan
    iteration = 0

    while True:
        new_V = bellman_update(V, R, gamma, p_intended, p_side)
        delta = np.nanmax(np.abs(new_V - V))
        V = new_V
        iteration += 1
        if delta < theta:
            break

    policy = extract_policy(V, R, gamma, p_intended, p_side)

    print(f"\n=== {label} ===")
    print(f"gamma={gamma}, p_intended={p_intended}, p_side={p_side}, "
          f"living_reward={living_reward} -> converged in {iteration} iterations")
    print(np.round(V, 3))

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

    if plot:
        plt.figure(figsize=(6, 4))
        plt.title(f"{label}\n(gamma={gamma}, p_intended={p_intended}, living_reward={living_reward})")
        im = plt.imshow(V, cmap="coolwarm", interpolation="nearest")

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
        plt.tight_layout()

    return V, policy


# ---------------------------------------------------------------------------
# Experimental settings
# ---------------------------------------------------------------------------
experiments = [
    # Baseline
    dict(gamma=0.9, p_intended=0.8, p_side=0.1, living_reward=-0.01,
         label="Baseline"),

     # 1. Discount factor variations
   # dict(gamma=0.5, p_intended=0.8, p_side=0.1, living_reward=-0.01,
   #      label="Myopic agent (gamma=0.5)"),
   # dict(gamma=0.99, p_intended=0.8, p_side=0.1, living_reward=-0.01,
   #      label="Far-sighted agent (gamma=0.99)"),
   # dict(gamma=0.1, p_intended=0.8, p_side=0.1, living_reward=-0.01,
   #      label="Very myopic agent (gamma=0.1)"),
 
    # 2. Stochasticity level variations
  #  dict(gamma=0.9, p_intended=1.0, p_side=0.0, living_reward=-0.01,
  #       label="Deterministic transitions"),
  #  dict(gamma=0.9, p_intended=0.6, p_side=0.2, living_reward=-0.01,
  #       label="High stochasticity"),
  #  dict(gamma=0.9, p_intended=0.34, p_side=0.33, living_reward=-0.01,
  #       label="Near-random transitions"),

    # 3. Living reward variations
 #   dict(gamma=0.9, p_intended=0.8, p_side=0.1, living_reward=-0.04,
 #        label="Low living cost"),
 #   dict(gamma=0.9, p_intended=0.8, p_side=0.1, living_reward=-2.0,
 #            label="High cost"),
 #   dict(gamma=0.9, p_intended=0.8, p_side=0.1, living_reward=0.0,
 #        label="No living cost"),

     # 4. Challenge: "The Risky Shortcut"
    # The trap sits next to the only short path to the goal.
    # You must find settings where the agent (a) avoids it via detour
    # and (b) risks walking past it. No single parameter alone determines this —
    # it's the interaction of p_side, gamma, living_reward, and reward magnitudes.
    dict(gamma=0.9, p_intended=0.8, p_side=0.1, living_reward=-0.01,
         label="Challenge: cautious or reckless? (tune me!)"),


]

# ---------------------------------------------------------------------------
# CHALLENGE 
# ---------------------------------------------------------------------------
# Using the SAME grid (trap adjacent to the shortest path to goal), find two
# distinct parameter settings such that:
#
#   (A) The optimal policy takes the LONG way around the trap
#       (high risk-aversion regime)
#   (B) The optimal policy walks directly past/near the trap to save time
#       (risk-tolerant regime)
#
# You may change: gamma, p_intended/p_side, living_reward, and reward
# magnitudes for goal/trap (edit run_experiment's R assignment if needed).
#
# Questions to answer in your write-up:
#   1. Which single parameter had the LARGEST effect on switching between
#      (A) and (B)? Why does that make sense mathematically in the Bellman
#      equation?
#   2. Can you find a setting where changing gamma ALONE flips the policy,
#      holding p_side and living_reward fixed? What does that tell you about
#      how discounting affects risk sensitivity in stochastic environments?
#   3. What happens if p_side is very high (e.g., 0.3) but living_reward is
#      very negative (e.g., -5.0)? Which effect wins, and why might that be
#      counter-intuitive?
# ---------------------------------------------------------------------------



if __name__ == "__main__":
    results = {}
    for exp in experiments:
        exp = dict(exp)  # shallow copy so we don't mutate the list
        label = exp.pop("label")
        results[label] = run_experiment(label=label, **exp)

    plt.show()