"""
Test and demo script for asset allocation RL
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from config import CONFIG_MVP_SANITY, CONFIG_MVP_MULTIPERIOD, CONFIG_TWO_ASSETS, TRAINING_CONFIG
from env import AssetAllocationEnv
from agent import PPOAgent
from utils import merton_optimal_allocation, project_action_to_feasible_set
import torch


def test_environment():
    """Test environment logic without RL."""
    print("=" * 60)
    print("TEST 1: Environment Dynamics")
    print("=" * 60)

    env = AssetAllocationEnv(**CONFIG_MVP_SANITY)
    obs, _ = env.reset()

    print(f"Initial observation: {obs}")
    print(f"  Normalized time: {obs[0]:.4f}")
    print(f"  Normalized wealth: {obs[1]:.4f}")
    print(f"  Portfolio: {obs[2:]}")

    # Take one step with zero action (no adjustment)
    action = np.zeros(env.n_total_assets)
    obs, reward, done, truncated, info = env.step(action)

    print(f"\nAfter one step (zero action):")
    print(f"  Reward: {reward:.6f}")
    print(f"  New wealth: {info['wealth']:.4f}")
    print(f"  New portfolio: {info['portfolio']}")
    print(f"  Portfolio return: {info['portfolio_return']:.4f}")
    print(f"  Done: {done}")


def test_action_projection():
    """Test action projection to feasible set."""
    print("\n" + "=" * 60)
    print("TEST 2: Action Projection")
    print("=" * 60)

    portfolio = np.array([0.5, 0.5])
    max_adj = 0.1

    # Test case 1: Large action
    action = np.array([0.5, -0.5])
    projected = project_action_to_feasible_set(action, portfolio, max_adj)

    print(f"Original action: {action}")
    print(f"Projected action: {projected}")
    print(f"New portfolio: {portfolio + projected}")
    print(f"Sum of action: {np.sum(projected):.6f} (should be ~0)")

    # Test case 2: Action causing negative portfolio
    action = np.array([-0.8, 0.1])
    projected = project_action_to_feasible_set(action, portfolio, max_adj)

    print(f"\nOriginal action (would short): {action}")
    print(f"Projected action: {projected}")
    print(f"New portfolio: {portfolio + projected}")
    print(f"All non-negative: {np.all(portfolio + projected >= -1e-6)}")


def test_analytical_solution():
    """Test analytical solution for sanity config."""
    print("\n" + "=" * 60)
    print("TEST 3: Analytical Solution (Merton)")
    print("=" * 60)

    result = merton_optimal_allocation(
        a=CONFIG_MVP_SANITY["a"][0],
        r=CONFIG_MVP_SANITY["r"],
        s=CONFIG_MVP_SANITY["s"][0],
        gamma=CONFIG_MVP_SANITY["gamma"],
    )

    print(f"Parameters:")
    print(f"  a (expected return): {CONFIG_MVP_SANITY['a'][0]:.4f}")
    print(f"  r (risk-free rate): {CONFIG_MVP_SANITY['r']:.4f}")
    print(f"  s (std dev): {CONFIG_MVP_SANITY['s'][0]:.4f}")
    print(f"  γ (risk aversion): {CONFIG_MVP_SANITY['gamma']:.4f}")

    print(f"\nOptimal allocation (unconstrained):")
    print(f"  Cash: {result['p_cash']:.4f}")
    print(f"  Risk asset: {result['p_risky']:.4f}")
    print(f"  Feasible: {result['is_valid']}")


def test_ppo_agent():
    """Test PPO agent initialization and action selection."""
    print("\n" + "=" * 60)
    print("TEST 4: PPO Agent")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = AssetAllocationEnv(**CONFIG_MVP_SANITY)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=TRAINING_CONFIG["learning_rate"],
        gamma=TRAINING_CONFIG["gamma_discount"],
        device=device,
    )

    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"Device: {device}")

    # Test action selection
    obs, _ = env.reset()
    action = agent.select_action(obs)

    print(f"\nInitial observation: {obs}")
    print(f"Agent action: {action}")
    print(f"Action in [-1, 1]: {np.all(action >= -1) and np.all(action <= 1)}")


def test_quick_train():
    """Quick training test (100 episodes)."""
    print("\n" + "=" * 60)
    print("TEST 5: Quick Training (100 episodes)")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = AssetAllocationEnv(**CONFIG_MVP_SANITY)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=TRAINING_CONFIG["learning_rate"],
        gamma=TRAINING_CONFIG["gamma_discount"],
        device=device,
    )

    np.random.seed(TRAINING_CONFIG["seed"])
    torch.manual_seed(TRAINING_CONFIG["seed"])

    rewards = []
    for episode in range(100):
        obs, _ = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action = agent.select_action_stochastic(obs)
            obs, reward, done, _, _ = env.step(action)
            episode_reward += reward

        rewards.append(episode_reward)

    print(f"Completed 100 episodes")
    print(f"Rewards (first 10): {rewards[:10]}")
    print(f"Rewards (last 10): {rewards[-10:]}")
    print(f"Average reward (last 10): {np.mean(rewards[-10:]):.6f}")


if __name__ == "__main__":
    # Run all tests
    test_environment()
    test_action_projection()
    test_analytical_solution()
    test_ppo_agent()
    test_quick_train()

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
