import random

# --- Define states ---
states = ["Sunny", "Rainy", "Cloudy"]

# --- Transition probabilities (rows must sum to 1) ---
transitions = {
    "Sunny":  [0.6, 0.2, 0.2],  # Sunny -> (Sunny, Rainy, Cloudy)
    "Rainy":  [0.0, 0.5, 0.5],  # Rainy -> (Sunny, Rainy, Cloudy)
    "Cloudy": [0.5, 0.2, 0.3]   # Cloudy -> (Sunny, Rainy, Cloudy)
}

# --- Function to get next state ---
def next_state(current_state):
    """
    TODO:
    1. Use random.choices() with the transition probabilities
    2. Return the chosen next state
    """
    pass

# --- Main simulation ---
def run_simulation(start_state, steps=10):
    state = start_state
    sequence = [state]

    for _ in range(steps):
        # TODO: call next_state() and update `state`
        # Append result to sequence
        pass

    return sequence

# --- Run ---
if __name__ == "__main__":
    result = run_simulation("Cloudy", steps=10)
    print("Weather sequence:", result)