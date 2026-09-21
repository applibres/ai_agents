import gymnasium as gym
import numpy as np
from collections import defaultdict
import random
import matplotlib.pyplot as plt

# Parameters
num_episodes = 5000
gamma = 0.9
seed = 40
random.seed(seed)
np.random.seed(seed)

# Environment
env = gym.make("FrozenLake-v1", is_slippery=True, render_mode="ansi")
env.action_space.seed(seed)
env.reset(seed=seed)

def random_policy(state):
    return env.action_space.sample()

def generate_episode(env, policy):
    episode = []
    state, _ = env.reset()
    done = False
    while not done:
        action = policy(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        episode.append((state, action, reward))
        state = next_state
        done = terminated or truncated
    return episode

def mc_first_visit_incremental(env, num_episodes, gamma=0.9):
    V = defaultdict(float)
    N = defaultdict(int)

    for ep in range(num_episodes):
        episode = generate_episode(env, random_policy)
        visited_states = set()
        G = 0
        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            G = gamma * G + reward
            if state not in visited_states:
                N[state] += 1
                V[state] += (1 / N[state]) * (G - V[state])
                visited_states.add(state)
    return V

# Run MC Evaluation
V = mc_first_visit_incremental(env, num_episodes)
print("Estimated State Values:")
for s, v in sorted(V.items()):
    print(f"State {s}: {v:.3f}")
