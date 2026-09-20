"""Naive DQN (no replay buffer, no target network) on CartPole-v1 with Gymnasium."""
import random
import os
# Must be set before numpy/torch load to avoid duplicate OpenMP runtime crash on macOS
# (conda's numpy/llvm-openmp and torch's bundled libiomp5 both try to init OpenMP).
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

torch.set_num_threads(1)


GAMMA = 0.99
LR = 1e-3
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY_STEPS = 20000
NUM_STEPS = 50000
HIDDEN_SIZE = 128


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def select_action(q_net: QNetwork, state: np.ndarray, epsilon: float, n_actions: int) -> int:
    if random.random() < epsilon:
        return random.randrange(n_actions)
    with torch.no_grad():
        state_t = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        q_values = q_net(state_t)
        return int(torch.argmax(q_values, dim=1).item())


def train():
    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    q_net = QNetwork(obs_dim, n_actions)
    optimizer = optim.Adam(q_net.parameters(), lr=LR)

    episode_rewards = []
    episode_reward = 0.0
    state, _ = env.reset()

    i = 0
    while i < NUM_STEPS:
        epsilon = max(
            EPSILON_END,
            EPSILON_START - (EPSILON_START - EPSILON_END) * i / EPSILON_DECAY_STEPS,
        )

        action = select_action(q_net, state, epsilon, n_actions)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        episode_reward += reward
        i += 1

        state_t = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        next_state_t = torch.as_tensor(next_state, dtype=torch.float32).unsqueeze(0)

        # TD target: bootstrapped from the same online network (no target net) -> "naive" DQN
        with torch.no_grad():
            if done:
                target = torch.tensor([reward], dtype=torch.float32)
            else:
                next_q_max = torch.max(q_net(next_state_t), dim=1)[0]
                target = reward + GAMMA * next_q_max

        q_values = q_net(state_t)
        q_action = q_values.gather(1, torch.tensor([[action]])).squeeze(1)

        loss = (target - q_action).pow(2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        state = next_state

        if done:
            episode_rewards.append(episode_reward)
            episode_reward = 0.0
            state, _ = env.reset()

            avg_reward = np.mean(episode_rewards[-20:])
            if len(episode_rewards) % 10 == 0:
                print(
                    f"Step {i}/{NUM_STEPS} episode={len(episode_rewards)} "
                    f"reward={episode_rewards[-1]:.1f} avg20={avg_reward:.1f} epsilon={epsilon:.3f}"
                )

    env.close()
    return q_net


def run_trained_agent(q_net: QNetwork, episodes: int = 5):
    env = gym.make("CartPole-v1", render_mode="human")
    n_actions = env.action_space.n

    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action = select_action(q_net, state, epsilon=0.0, n_actions=n_actions)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
        print(f"[Render] Episode {ep + 1} reward={total_reward:.1f}")

    env.close()


if __name__ == "__main__":
    trained_q_net = train()
    run_trained_agent(trained_q_net)
