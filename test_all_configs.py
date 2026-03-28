"""
Train and evaluate RL agent on all configurations in config.py

This script runs training on each predefined configuration and reports performance metrics.
Enhanced version with random configuration generation.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import time
from tabulate import tabulate
import json
from datetime import datetime

from config import (
    CONFIG_MVP_SANITY,
    CONFIG_MVP_MULTIPERIOD,
    CONFIG_TWO_ASSETS,
    CONFIG_THREE_ASSETS,
    CONFIG_FOUR_ASSETS,
    CONFIG_CORRELATED_ASSETS,
    TRAINING_CONFIG
)
from train import train


def format_config_name(config_dict):
    """Generate a readable name for a configuration."""
    name_parts = []

    # Extract key characteristics
    n_assets = config_dict.get("n_assets", 0)
    T = config_dict.get("T", 0)

    # Check for special features
    if "cov_matrix" in config_dict:
        name_parts.append(f"Correlated(n={n_assets})")
    elif n_assets == 1:
        if T == 1:
            name_parts.append("MVP_Sanity")
        else:
            name_parts.append(f"Multiperiod(T={T})")
    else:
        name_parts.append(f"n={n_assets}_T={T}")

    return "_".join(name_parts)


def generate_random_config(seed=None, n_assets=None, T=None):
    """
    Generate a random configuration that meets assignment requirements.

    Assignment requirements:
    - n > 2 (but n < 5, so n can be 3 or 4)
    - X(0) = 1 (initial wealth = 1.0)
    - One period return ~ N(a(k), s(k)) where s(k) is variance
    - Initial portfolio p(k) with p(0) cash, sum(p) = 1
    - Cash interest rate r
    - At most adjust 10% per period (one-way turnover constraint)
    - Time horizon T < 10

    Args:
        seed: Random seed for reproducibility
        n_assets: Number of risk assets (3 or 4). If None, randomly choose.
        T: Time horizon (1-9). If None, randomly choose.

    Returns:
        Random configuration dictionary
    """
    rng = np.random.RandomState(seed)

    # Randomly choose n_assets if not specified (3 or 4 to satisfy n > 2 and n < 5)
    if n_assets is None:
        n_assets = rng.choice([3, 4])

    # Randomly choose T if not specified (1-9 to satisfy T < 10)
    if T is None:
        T = rng.randint(1, 10)  # 1 to 9 inclusive

    # Cash interest rate: realistic range 1-4%
    r = rng.uniform(0.01, 0.04)

    # Expected returns: above risk-free rate with varying risk premiums
    # Higher returns for potentially riskier assets
    risk_premiums = rng.uniform(0.02, 0.12, size=n_assets)
    a = r + risk_premiums

    # Variances: make higher return assets have higher variance (positive correlation)
    base_vol = rng.uniform(0.05, 0.15, size=n_assets)
    # Scale volatility by risk premium to create positive risk-return relationship
    vol_scaling = 0.5 + 0.5 * (risk_premiums / np.max(risk_premiums))
    std_devs = base_vol * vol_scaling
    s = std_devs ** 2  # Convert to variance

    # Risk aversion: reasonable range
    gamma = rng.uniform(0.8, 2.5)

    # Initial portfolio: ensure diversification
    # Generate random weights with bias towards cash
    weights = rng.random(n_assets + 1)
    # Cash bias: ensure some cash allocation but not too much
    weights[0] = rng.uniform(0.1, 0.5)
    # Random allocation among risky assets
    weights[1:] = rng.random(n_assets)
    initial_portfolio = weights / weights.sum()

    # Occasionally add correlation matrix (30% chance)
    config = {
        "n_assets": n_assets,
        "T": T,
        "r": r,
        "a": a.tolist(),
        "s": s.tolist(),
        "gamma": gamma,
        "initial_portfolio": initial_portfolio.tolist(),
        "initial_wealth": 1.0,
        "max_portfolio_adjustment": 0.1,
    }

    # Add correlation matrix with 30% probability
    if rng.random() < 0.3 and n_assets >= 2:
        # Generate random correlation matrix
        corr = np.eye(n_assets)
        for i in range(n_assets):
            for j in range(i+1, n_assets):
                # Generate random correlation between -0.3 and 0.7
                rho = rng.uniform(-0.3, 0.7)
                corr[i, j] = rho
                corr[j, i] = rho

        # Convert correlation to covariance matrix
        std_matrix = np.diag(std_devs)
        cov_matrix = std_matrix @ corr @ std_matrix

        config["cov_matrix"] = cov_matrix.tolist()

    return config


def generate_random_configs(num_configs=5, base_seed=12345):
    """
    Generate multiple random configurations for comprehensive testing.

    Args:
        num_configs: Number of random configurations to generate
        base_seed: Base random seed for reproducibility

    Returns:
        List of (name, config) tuples
    """
    random_configs = []

    # Ensure diversity in n and T values
    n_options = [3, 4]  # n must be > 2 and < 5
    t_options = list(range(1, 10))  # T < 10

    for i in range(num_configs):
        # Use different seed for each config
        seed = base_seed + i * 100

        # Vary n and T to cover different scenarios
        if i < len(n_options):
            n_assets = n_options[i % len(n_options)]
        else:
            n_assets = None  # Random choice

        if i < len(t_options):
            T = t_options[i % len(t_options)]
        else:
            T = None  # Random choice

        config = generate_random_config(seed=seed, n_assets=n_assets, T=T)

        # Create descriptive name
        n = config["n_assets"]
        T_val = config["T"]
        has_corr = "cov_matrix" in config
        corr_str = "_corr" if has_corr else ""

        name = f"Random_{i+1}_n{n}_T{T_val}{corr_str}"

        random_configs.append((name, config))

    return random_configs


def train_and_evaluate(config, config_name, train_config, episodes_per_config=500):
    """
    Train agent on a specific configuration and evaluate performance.

    Args:
        config: Environment configuration dictionary
        config_name: Name of the configuration
        train_config: Training hyperparameters
        episodes_per_config: Number of training episodes

    Returns:
        Dictionary with evaluation metrics
    """
    print(f"\n{'='*60}")
    print(f"Training on configuration: {config_name}")
    print(f"{'='*60}")

    # Print configuration summary
    print(f"n_assets: {config['n_assets']}")
    print(f"T (time steps): {config['T']}")
    print(f"Risk-free rate (r): {config['r']:.3f}")
    print(f"Expected returns (a): {[f'{x:.3f}' for x in config['a']]}")
    print(f"Variances (s): {[f'{x:.6f}' for x in config['s']]}")
    print(f"Risk aversion (γ): {config['gamma']:.2f}")
    print(f"Initial portfolio: {[f'{x:.2f}' for x in config['initial_portfolio']]}")

    if "cov_matrix" in config:
        print("Using correlation matrix")

    # Create a copy of training config with adjusted episode count
    current_train_config = train_config.copy()
    current_train_config["n_episodes"] = episodes_per_config

    # Track training time
    start_time = time.time()

    try:
        # Train agent
        agent, episode_rewards, episode_final_wealths = train(
            env_config=config,
            train_config=current_train_config,
            n_episodes=episodes_per_config
        )

        training_time = time.time() - start_time

        # Compute performance metrics
        if len(episode_rewards) > 0:
            # Last 100 episodes performance (convergence)
            last_n = min(100, len(episode_rewards))
            recent_rewards = episode_rewards[-last_n:]
            recent_wealths = episode_final_wealths[-last_n:]

            avg_reward = np.mean(recent_rewards)
            std_reward = np.std(recent_rewards)
            avg_wealth = np.mean(recent_wealths)
            std_wealth = np.std(recent_wealths)

            # Overall training performance
            overall_avg_reward = np.mean(episode_rewards)
            overall_avg_wealth = np.mean(episode_final_wealths)

            # Check for learning (compare first vs last 100 episodes)
            first_n = min(100, len(episode_rewards))
            if len(episode_rewards) >= 200:
                first_rewards = episode_rewards[:first_n]
                last_rewards = episode_rewards[-first_n:]
                improvement = np.mean(last_rewards) - np.mean(first_rewards)
            else:
                improvement = 0.0

            # Get final policy action for analysis
            from env import AssetAllocationEnv
            env = AssetAllocationEnv(**config)
            obs, _ = env.reset()
            final_action = agent.select_action(obs)
            final_action_scaled = final_action * config.get("max_portfolio_adjustment", 0.1)

            metrics = {
                "config_name": config_name,
                "training_success": True,
                "training_time": training_time,
                "episodes_trained": episodes_per_config,
                "avg_recent_reward": avg_reward,
                "std_recent_reward": std_reward,
                "avg_recent_wealth": avg_wealth,
                "std_recent_wealth": std_wealth,
                "overall_avg_reward": overall_avg_reward,
                "overall_avg_wealth": overall_avg_wealth,
                "reward_improvement": improvement,
                "final_action": final_action.tolist(),
                "final_action_scaled": final_action_scaled.tolist(),
                "n_assets": config["n_assets"],
                "T": config["T"],
                "has_correlation": "cov_matrix" in config,
            }

            print(f"\nTraining completed in {training_time:.2f} seconds")
            print(f"Recent performance (last {last_n} episodes):")
            print(f"  Avg reward: {avg_reward:.6f} ± {std_reward:.6f}")
            print(f"  Avg final wealth: {avg_wealth:.4f} ± {std_wealth:.4f}")
            print(f"  Reward improvement: {improvement:.6f}")
            print(f"Final action (scaled): {[f'{x:.4f}' for x in final_action_scaled]}")

        else:
            metrics = {
                "config_name": config_name,
                "training_success": False,
                "error": "No rewards collected",
                "training_time": training_time,
            }
            print(f"Training failed: No rewards collected")

    except Exception as e:
        training_time = time.time() - start_time
        metrics = {
            "config_name": config_name,
            "training_success": False,
            "error": str(e),
            "training_time": training_time,
        }
        print(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()

    return metrics


def main():
    """Main function to train on all configurations."""
    print("\n" + "="*70)
    print("RL Asset Allocation - Multi-Configuration Training (Enhanced)")
    print("="*70)
    print("Includes 6 predefined configurations + 5 random configurations")
    print("Random configurations satisfy: n > 2, n < 5, T < 10")
    print("="*70)

    # Set random seeds for reproducibility
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Define predefined configurations
    predefined_configs = [
        ("MVP_Sanity_T1", CONFIG_MVP_SANITY),
        ("Multiperiod_T5", CONFIG_MVP_MULTIPERIOD),
        ("Two_Assets_T5", CONFIG_TWO_ASSETS),
        ("Three_Assets_T7", CONFIG_THREE_ASSETS),
        ("Four_Assets_T9", CONFIG_FOUR_ASSETS),
        ("Correlated_T5", CONFIG_CORRELATED_ASSETS),
    ]

    # Generate 5 random configurations
    print("\nGenerating 5 random configurations...")
    random_configs = generate_random_configs(num_configs=5, base_seed=12345)

    # Print random config summaries
    for name, config in random_configs:
        n = config["n_assets"]
        T = config["T"]
        has_corr = "cov_matrix" in config
        print(f"  {name}: n={n}, T={T}, γ={config['gamma']:.2f}, "
              f"r={config['r']:.3f}, {'with corr' if has_corr else 'no corr'}")

    # Combine all configurations
    all_configs = predefined_configs + random_configs

    # Adjust training parameters for faster testing
    train_config = TRAINING_CONFIG.copy()
    train_config["n_episodes"] = 500  # Reduced for quicker testing
    train_config["learning_rate"] = 3e-4
    train_config["use_lr_schedule"] = True  # Enable LR scheduling

    # Store all results
    all_results = []

    # Train on each configuration
    for config_name, config in all_configs:
        metrics = train_and_evaluate(
            config=config,
            config_name=config_name,
            train_config=train_config,
            episodes_per_config=500
        )
        all_results.append(metrics)

    # Print summary table
    print("\n" + "="*70)
    print("SUMMARY TABLE - All Configurations")
    print("="*70)
    print("Predefined (1-6) + Random (7-11)")
    print("="*70)

    # Prepare table data
    table_data = []
    for i, result in enumerate(all_results):
        config_type = "P" if i < 6 else "R"
        config_num = i + 1

        if result["training_success"]:
            row = [
                f"{config_num} ({config_type})",
                result["config_name"],
                f"{result['avg_recent_reward']:.6f}",
                f"{result['std_recent_reward']:.6f}",
                f"{result['avg_recent_wealth']:.4f}",
                f"{result['std_recent_wealth']:.4f}",
                f"{result['reward_improvement']:.6f}",
                f"{result['training_time']:.1f}s",
                "✓",
            ]
        else:
            row = [
                f"{config_num} ({config_type})",
                result["config_name"],
                "N/A",
                "N/A",
                "N/A",
                "N/A",
                "N/A",
                f"{result['training_time']:.1f}s",
                f"✗ ({result.get('error', 'Unknown')[:20]}...)",
            ]
        table_data.append(row)

    headers = [
        "# (Type)",
        "Configuration",
        "Avg Reward",
        "Std Reward",
        "Avg Wealth",
        "Std Wealth",
        "Improvement",
        "Time",
        "Status",
    ]

    print(tabulate(table_data, headers=headers, tablefmt="grid"))

    # Group statistics
    print("\n" + "="*70)
    print("GROUP STATISTICS")
    print("="*70)

    # Predefined vs Random comparison
    predefined_results = all_results[:6]
    random_results = all_results[6:]

    def compute_group_stats(results):
        if not results:
            return None

        successful = [r for r in results if r["training_success"]]
        if not successful:
            return None

        avg_rewards = [r["avg_recent_reward"] for r in successful if "avg_recent_reward" in r]
        avg_wealths = [r["avg_recent_wealth"] for r in successful if "avg_recent_wealth" in r]

        return {
            "count": len(results),
            "success_count": len(successful),
            "success_rate": len(successful) / len(results) if len(results) > 0 else 0,
            "avg_reward": np.mean(avg_rewards) if avg_rewards else None,
            "avg_wealth": np.mean(avg_wealths) if avg_wealths else None,
        }

    predefined_stats = compute_group_stats(predefined_results)
    random_stats = compute_group_stats(random_results)

    if predefined_stats:
        print(f"Predefined Configurations (n=6):")
        print(f"  Success rate: {predefined_stats['success_rate']:.1%} ({predefined_stats['success_count']}/{predefined_stats['count']})")
        if predefined_stats['avg_reward'] is not None:
            print(f"  Average reward: {predefined_stats['avg_reward']:.6f}")
        if predefined_stats['avg_wealth'] is not None:
            print(f"  Average wealth: {predefined_stats['avg_wealth']:.4f}")

    if random_stats:
        print(f"\nRandom Configurations (n={random_stats['count']}):")
        print(f"  Success rate: {random_stats['success_rate']:.1%} ({random_stats['success_count']}/{random_stats['count']})")
        if random_stats['avg_reward'] is not None:
            print(f"  Average reward: {random_stats['avg_reward']:.6f}")
        if random_stats['avg_wealth'] is not None:
            print(f"  Average wealth: {random_stats['avg_wealth']:.4f}")

    # Save results to JSON file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"training_results_enhanced_{timestamp}.json"

    # Convert numpy arrays to lists for JSON serialization
    json_results = []
    for result in all_results:
        json_result = {}
        for key, value in result.items():
            if isinstance(value, np.ndarray):
                json_result[key] = value.tolist()
            elif isinstance(value, np.generic):
                json_result[key] = value.item()
            else:
                json_result[key] = value
        json_results.append(json_result)

    with open(results_file, 'w') as f:
        json.dump(json_results, f, indent=2, default=str)

    print(f"\nResults saved to: {results_file}")

    # Overall statistics
    successful = sum(1 for r in all_results if r["training_success"])
    total = len(all_results)

    print(f"\nOverall: {successful}/{total} configurations trained successfully")

    if successful == total:
        print("✓ All configurations completed successfully!")
    else:
        print("✗ Some configurations failed. Check logs for details.")

    # Check assignment requirements
    print("\n" + "="*70)
    print("ASSIGNMENT REQUIREMENTS CHECK")
    print("="*70)

    # Check n > 2 for random configs (should all be 3 or 4)
    random_n_values = [config["n_assets"] for _, config in random_configs]
    n_gt_2 = all(n > 2 for n in random_n_values)
    n_lt_5 = all(n < 5 for n in random_n_values)

    # Check T < 10
    random_T_values = [config["T"] for _, config in random_configs]
    T_lt_10 = all(T < 10 for T in random_T_values)

    print(f"Random configs satisfy n > 2: {'✓' if n_gt_2 else '✗'} (n values: {random_n_values})")
    print(f"Random configs satisfy n < 5: {'✓' if n_lt_5 else '✗'} (n values: {random_n_values})")
    print(f"Random configs satisfy T < 10: {'✓' if T_lt_10 else '✗'} (T values: {random_T_values})")

    if n_gt_2 and n_lt_5 and T_lt_10:
        print("\n✓ All assignment requirements satisfied for random configurations!")
    else:
        print("\n✗ Some assignment requirements not satisfied.")

    return all_results


if __name__ == "__main__":
    # Install tabulate if not available: pip install tabulate
    try:
        from tabulate import tabulate
    except ImportError:
        print("Error: 'tabulate' package not installed.")
        print("Install with: pip install tabulate")
        sys.exit(1)

    main()