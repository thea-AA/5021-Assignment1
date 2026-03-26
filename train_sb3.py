"""
Training using stable-baselines3 PPO (more robust)
"""
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt

from config import CONFIG_MVP_SANITY, CONFIG_MVP_MULTIPERIOD, CONFIG_TWO_ASSETS
from env import AssetAllocationEnv
from utils import merton_optimal_allocation


class RewardLogger(BaseCallback):
    """Log rewards during training."""

    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self.episode_counter = 0

    def _on_step(self) -> bool:
        if "dones" in self.locals:
            dones = self.locals["dones"]
            if np.any(dones):
                for i, done in enumerate(dones):
                    if done:
                        reward = self.locals.get("rewards", [0])[i] if isinstance(
                            self.locals.get("rewards", [0]), (list, np.ndarray)
                        ) else 0
                        self.episode_rewards.append(reward)
                        self.episode_counter += 1
                        if self.episode_counter % 500 == 0:
                            avg_reward = np.mean(self.episode_rewards[-500:])
                            print(
                                f"Episode {self.episode_counter}: Avg Reward (last 500) = {avg_reward:.6f}"
                            )
        return True


def train_sb3(config, n_steps=100000):
    """
    Train PPO agent using stable-baselines3.

    Args:
        config: Environment configuration
        n_steps: Number of training steps
    """
    print("Creating environment...")
    env = AssetAllocationEnv(**config)

    print("Creating PPO agent...")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        verbose=0,
    )

    print(f"Training for {n_steps} steps...")
    callback = RewardLogger()
    model.learn(total_timesteps=n_steps, callback=callback)

    return model, callback.episode_rewards


def evaluate_policy(model, config, n_episodes=100):
    """
    Evaluate learned policy.
    """
    env = AssetAllocationEnv(**config)
    episode_returns = []
    episode_final_wealths = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        episode_return = 0

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, _, _ = env.step(action)
            episode_return += reward

        episode_returns.append(episode_return)
        episode_final_wealths.append(env.wealth)

    return {
        "mean_return": np.mean(episode_returns),
        "std_return": np.std(episode_returns),
        "mean_wealth": np.mean(episode_final_wealths),
        "std_wealth": np.std(episode_final_wealths),
    }


def compare_with_analytical(config, model):
    """
    Compare RL policy with analytical solution for sanity checks.
    """
    if config["n_assets"] != 1:
        print("Analytical comparison only for n_assets=1")
        return

    print("\n" + "=" * 60)
    print("COMPARISON: RL vs Analytical Solution")
    print("=" * 60)

    # Analytical solution
    result = merton_optimal_allocation(
        a=config["a"][0],
        r=config["r"],
        s=config["s"][0],
        gamma=config["gamma"],
    )

    print(f"\nParameters:")
    print(f"  Expected return (a): {config['a'][0]:.4f}")
    print(f"  Risk-free rate (r): {config['r']:.4f}")
    print(f"  Std dev (s): {config['s'][0]:.4f}")
    print(f"  Risk aversion (γ): {config['gamma']:.4f}")

    print(f"\nAnalytical Optimal (unconstrained Merton):")
    print(f"  Cash: {result['p_cash']:.4f}")
    print(f"  Risk asset: {result['p_risky']:.4f}")
    print(f"  Feasible: {result['is_valid']}")

    # RL policy
    env = AssetAllocationEnv(**config)
    obs, _ = env.reset()
    action, _ = model.predict(obs, deterministic=True)

    print(f"\nRL Learned Action:")
    print(f"  Raw action: {action}")
    print(f"  Scaled adjustment (10% max): {action * 0.1}")
    print(f"  Initial portfolio: {env.portfolio}")
    new_portfolio = env.portfolio + np.clip(action * 0.1, -0.1, 0.1)
    new_portfolio = new_portfolio / np.sum(new_portfolio)
    print(f"  Adjusted portfolio: {new_portfolio}")


if __name__ == "__main__":
    print("Training on MVP Sanity Check (n=1, T=1)...")
    model, rewards = train_sb3(CONFIG_MVP_SANITY, n_steps=50000)

    print("\n" + "=" * 60)
    print("TRAINING RESULTS")
    print("=" * 60)
    print(f"Total episodes trained: {len(rewards)}")
    print(f"Mean episode reward (last 100): {np.mean(rewards[-100:]):.6f}")
    print(f"Std episode reward (last 100): {np.std(rewards[-100:]):.6f}")

    # Evaluation
    print("\nEvaluating policy on 100 episodes...")
    eval_results = evaluate_policy(model, CONFIG_MVP_SANITY, n_episodes=100)
    print(f"\nEvaluation Results:")
    print(f"  Mean return: {eval_results['mean_return']:.6f}")
    print(f"  Std return: {eval_results['std_return']:.6f}")
    print(f"  Mean final wealth: {eval_results['mean_wealth']:.6f}")
    print(f"  Std final wealth: {eval_results['std_wealth']:.6f}")

    # Comparison
    compare_with_analytical(CONFIG_MVP_SANITY, model)

    # Save model
    model.save("asset_allocation_ppo")
    print("\nModel saved to asset_allocation_ppo.zip")

    # Plot learning curve
    if len(rewards) > 0:
        plt.figure(figsize=(10, 5))
        # Simple moving average
        window = 50
        smooth_rewards = np.convolve(rewards, np.ones(window) / window, mode="valid")
        plt.plot(range(len(smooth_rewards)), smooth_rewards, label="50-ep moving avg")
        plt.xlabel("Episode")
        plt.ylabel("Episode Reward")
        plt.title("Learning Curve: PPO on Asset Allocation")
        plt.legend()
        plt.grid()
        plt.savefig("learning_curve.png")
        print("Learning curve saved to learning_curve.png")
