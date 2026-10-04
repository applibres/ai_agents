"""Minimal CartPole-v1 exploration script: inspect spaces, reset/step outputs, and a random-action rollout."""
import os
# Must be set before numpy/torch load to avoid duplicate OpenMP runtime crash on macOS
# (conda's numpy/llvm-openmp and torch's bundled libiomp5 both try to init OpenMP).
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import gymnasium as gym

NUM_EPISODES = 3


def describe_env(env: gym.Env) -> None:
    print("=== Environment spec ===")
    print(f"Observation space: {env.observation_space}")
    print(f"  low:  {env.observation_space.low}")
    print(f"  high: {env.observation_space.high}")
    print(f"Action space: {env.action_space} (0=push left, 1=push right)")
    print(f"Reward range: {env.spec.reward_threshold if env.spec else 'n/a'}")
    print(f"Max episode steps: {env.spec.max_episode_steps if env.spec else 'n/a'}")
    print()


def run_episode(env: gym.Env, episode_idx: int) -> None:
    state, info = env.reset(seed=episode_idx)
    print(f"--- Episode {episode_idx} ---")
    print(f"Initial state (cart pos, cart vel, pole angle, pole ang vel): {state}")
    print(f"Reset info: {info}")

    total_reward = 0.0
    step = 0
    terminated = truncated = False
    while not (terminated or truncated):
        action = env.action_space.sample()
        state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step += 1
        if step <= 100:
            print(f"  step {step}: action={action} state={state} reward={reward} "
                  f"terminated={terminated} truncated={truncated}")

    print(f"Episode {episode_idx} finished after {step} steps, total reward={total_reward}")
    print(f"Ended due to: {'terminated (pole fell / cart out of bounds)' if terminated else 'truncated (step limit reached)'}")
    print()


def main():
    env = gym.make("CartPole-v1")
    describe_env(env)

    for episode in range(NUM_EPISODES):
        run_episode(env, episode)

    env.close()


if __name__ == "__main__":
    main()
