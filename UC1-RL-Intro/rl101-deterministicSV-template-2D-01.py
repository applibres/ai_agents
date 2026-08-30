import numpy as np

# Grid dimensions: 3 rows x 4 columns
ROWS, COLS = 3, 4

# Define rewards (default = 0 everywhere)
rewards = np.zeros((ROWS, COLS))

# Special states
trap = (1, 3)    # (row=1, col=3) in 0-indexed Python -> position (2,4) in human terms
goal = (0, 3)    # (row=0, col=3) -> position (1,4)
invalid = (1, 1)    # (row=1, col=1) -> position (2,2)

rewards[trap] = -1
rewards[goal] = 1


# Discount factor
gamma = 0.9

# Actions: up, down, left, right
actions = {
    """
    TODO: Implement a set of actions: Upper U, Down D,
    Rigth R, Left L.
    "U": (-1, 0),
    "D": (,),
    "L": (,),
    "R": (,)   
    """

    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1)
}

def is_valid_state(state):
    """Check if state is inside grid and not a invalid."""
    r, c = state
    if r < 0 or r >= ROWS or c < 0 or c >= COLS:
        return False
    if state == invalid:
        return False
    return True

def step(state, action):
    """Deterministic transition: apply action if valid, else stay in place."""
    r, c = state
    dr, dc = actions[action]
    new_state = (r + dr, c + dc)
    if is_valid_state(new_state):
        return new_state, rewards[new_state]
    else:
        return state, rewards[state]

# Initialize state-value function
V = np.zeros((ROWS, COLS))

# Mark invalid as None (not usable)
V[invalid] = None

def bellman_update(state, V, gamma):
    """Perform one Bellman update for a given state."""
    """
    TODO: Implement Bellman update 
    - return max value
    """
    raise NotImplementedError("bellman_update not implemented yet!")



# Perform iterative updates
def value_iteration(V, gamma, iterations=10):
    for it in range(iterations):
        new_V = V.copy()
        for r in range(ROWS):
            for c in range(COLS):
                state = (r, c)
                if state == invalid or state == goal or state == trap:
                    continue
                new_V[state] = bellman_update(state, V, gamma)
        V = new_V
        print(f"Iteration {it+1}:")
        print(V, "\n")
    return V

# Run the lab
final_values = value_iteration(V, gamma, iterations=10)

print("Final state-value function after 10 iterations:")
print(final_values)