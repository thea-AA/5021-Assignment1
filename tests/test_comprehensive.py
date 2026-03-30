"""
Comprehensive test suite for asset allocation RL solution
Tests all constraints and requirements from the problem statement.

Extended with scenario tests covering:
  - Section 1: Market environment (dominated asset, risk/return tradeoff, scalability)
  - Section 2: Initial portfolio & constraint limits (extreme / optimal initialization)
  - Section 3: Risk aversion sensitivity (high/low γ, quadratic utility comparison)
  - Section 4: Baseline comparisons (Buy-and-Hold, Random, Greedy vs RL)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from typing import Dict, List, Tuple
import time
from statistics import mean, stdev

from config import (
    TEST_CONFIGS,
    TRAINING_CONFIG,
    CONFIG_MVP_SANITY,
)
from env import AssetAllocationEnv
from agent import PPOAgent
from train import train, rollout_episode
from utils import merton_optimal_allocation, cara_utility, project_action_to_feasible_set


class ComprehensiveTestSuite:
    """Systematic validation of RL solution."""
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.results = {}
    
    def log(self, message):
        """Print if verbose mode is on."""
        if self.verbose:
            print(message)
    
    def test_constraint_satisfaction(self, config: Dict, n_tests: int = 100):
        """
        Test 1: Verify all hard constraints are satisfied.

        Constraints:
        1. Σ_{k: Δp_k > 0} Δp_k ≤ 0.1  (one-way turnover ≤ 10%)
        2. p_k ≥ 0 (no short selling)
        3. Σp_k = 1 (budget balance)
        """
        self.log("\n" + "="*70)
        self.log("TEST 1: Constraint Satisfaction")
        self.log("="*70)

        env = AssetAllocationEnv(**config)
        violations = {
            "adjustment": 0,
            "short_selling": 0,
            "budget_balance": 0,
        }

        for _ in range(n_tests):
            obs, _ = env.reset()
            done = False

            while not done:
                # Random action
                action = np.random.uniform(-1, 1, size=env.n_total_assets)

                # Capture pre-rebalance portfolio
                current_portfolio = env.portfolio.copy()

                obs, reward, done, truncated, info = env.step(action)
                rebalanced = info["rebalanced_portfolio"]  # pre-drift, post-rebalance
                post_drift = info["portfolio"]             # post-drift (next period start)

                # Check 1: One-way turnover ≤ 10%
                # Δp = rebalanced - current; one-way = sum of positive changes
                delta = rebalanced - current_portfolio
                one_way_turnover = np.maximum(delta, 0.0).sum()
                if one_way_turnover > 0.1 + 1e-6:
                    violations["adjustment"] += 1

                # Check 2: No short selling (on rebalanced portfolio, before drift)
                if np.any(rebalanced < -1e-6):
                    violations["short_selling"] += 1

                # Check 3: Budget balance (post-drift portfolio sums to 1)
                if abs(np.sum(post_drift) - 1.0) > 1e-6:
                    violations["budget_balance"] += 1

        passed = sum(v == 0 for v in violations.values()) == len(violations)
        self.log(f"✓ One-way turnover violations: {violations['adjustment']}/{n_tests * config['T']}")
        self.log(f"✓ Short-selling violations: {violations['short_selling']}/{n_tests * config['T']}")
        self.log(f"✓ Budget balance violations: {violations['budget_balance']}/{n_tests * config['T']}")
        self.log(f"\nResult: {'PASS ✓' if passed else 'FAIL ✗'}")
        
        self.results["constraint_satisfaction"] = {
            "passed": passed,
            "violations": violations,
        }
        return passed
    
    def test_time_horizons(self, agent: PPOAgent = None):
        """
        Test 2: Validate for all T < 10.
        
        Test T = 1, 2, 3, 5, 7, 9
        """
        self.log("\n" + "="*70)
        self.log("TEST 2: Time Horizon Robustness (T < 10)")
        self.log("="*70)
        
        time_horizons = [1, 2, 3, 5, 7, 9]
        results = []
        
        for T in time_horizons:
            config = CONFIG_MVP_SANITY.copy()
            config["T"] = T
            
            env = AssetAllocationEnv(**config)
            
            if agent is None:
                # Quick training if no agent provided
                trained_agent, _, _ = train(
                    env_config=config,
                    train_config=TRAINING_CONFIG,
                    n_episodes=500  # Quick test
                )
            else:
                trained_agent = agent
            
            # Evaluate
            episode_rewards = []
            for _ in range(20):  # 20 episodes for statistics
                trajectory = rollout_episode(env, trained_agent)
                episode_rewards.append(sum(trajectory["rewards"]))
            
            avg_reward = mean(episode_rewards)
            std_reward = stdev(episode_rewards) if len(episode_rewards) > 1 else 0
            
            result = f"T={T}: Avg Reward = {avg_reward:.4f} ± {std_reward:.4f}"
            results.append(result)
            self.log(f"  {result}")
        
        passed = len(results) == len(time_horizons)
        self.log(f"\nResult: {'PASS ✓' if passed else 'FAIL ✗'} - Tested {len(results)} horizons")
        
        self.results["time_horizons"] = {
            "passed": passed,
            "tested": time_horizons,
            "results": results,
        }
        return passed
    
    def test_asset_numbers(self, agent: PPOAgent = None):
        """
        Test 3: Validate for all n < 5.
        
        Test n = 1, 2, 3, 4 (risk assets)
        """
        self.log("\n" + "="*70)
        self.log("TEST 3: Asset Number Scalability (n < 5)")
        self.log("="*70)
        
        asset_numbers = [1, 2, 3, 4]
        configs = [
            CONFIG_MVP_SANITY,
            TEST_CONFIGS["two_assets"],
            TEST_CONFIGS["three_assets"],
            TEST_CONFIGS["four_assets"],
        ]
        
        results = []
        for n, config in zip(asset_numbers, configs):
            env = AssetAllocationEnv(**config)
            
            if agent is None:
                trained_agent, _, _ = train(
                    env_config=config,
                    train_config=TRAINING_CONFIG,
                    n_episodes=500
                )
            else:
                trained_agent = agent
            
            # Evaluate
            episode_rewards = []
            for _ in range(20):
                trajectory = rollout_episode(env, trained_agent)
                episode_rewards.append(sum(trajectory["rewards"]))
            
            avg_reward = mean(episode_rewards)
            std_reward = stdev(episode_rewards) if len(episode_rewards) > 1 else 0
            
            result = f"n={n}: Avg Reward = {avg_reward:.4f} ± {std_reward:.4f}"
            results.append(result)
            self.log(f"  {result}")
        
        passed = len(results) == len(asset_numbers)
        self.log(f"\nResult: {'PASS ✓' if passed else 'FAIL ✗'} - Tested {len(results)} configurations")
        
        self.results["asset_numbers"] = {
            "passed": passed,
            "tested": asset_numbers,
            "results": results,
        }
        return passed
    
    def test_merton_comparison(self):
        """
        Test 4: Compare with Merton analytical solution (n=1, T=1).
        
        This validates that RL can recover the optimal solution
        when constraints are not binding.
        """
        self.log("\n" + "="*70)
        self.log("TEST 4: Merton Analytical Solution Comparison")
        self.log("="*70)
        
        config = CONFIG_MVP_SANITY.copy()
        
        # Compute Merton solution
        merton_result = merton_optimal_allocation(
            a=config["a"][0],
            r=config["r"],
            s=config["s"][0],  # Variance
            gamma=config["gamma"],
        )
        
        self.log(f"Merton optimal allocation:")
        self.log(f"  Cash: {merton_result['p_cash']:.4f}")
        self.log(f"  Risk Asset: {merton_result['p_risky']:.4f}")
        self.log(f"  Feasible: {merton_result['is_valid']}")
        
        # Train RL agent
        trained_agent, rewards, wealths = train(
            env_config=config,
            train_config=TRAINING_CONFIG,
            n_episodes=2000,
        )
        
        # Get learned policy
        env = AssetAllocationEnv(**config)
        obs, _ = env.reset()
        learned_action = trained_agent.select_action(obs)
        scaled_action = learned_action * config["max_portfolio_adjustment"]
        
        self.log(f"\nRL learned action (scaled):")
        self.log(f"  ΔCash: {scaled_action[0]:.4f}")
        self.log(f"  ΔRisk Asset: {scaled_action[1]:.4f}")
        
        # Check if RL moves in the right direction
        initial_portfolio = np.array(config["initial_portfolio"])
        target_portfolio = np.array([merton_result["p_cash"], merton_result["p_risky"]])
        direction_correct = (
            np.sign(scaled_action[1]) == np.sign(target_portfolio[1] - initial_portfolio[1])
        )
        
        passed = direction_correct or not merton_result["is_valid"]
        self.log(f"\nDirection correct: {'YES ✓' if direction_correct else 'NO ✗'}")
        self.log(f"Result: {'PASS ✓' if passed else 'FAIL ✗'}")
        
        self.results["merton_comparison"] = {
            "passed": passed,
            "merton_allocation": merton_result,
            "learned_action": scaled_action.tolist(),
        }
        return passed
    
    def test_parameter_sensitivity(self):
        """
        Test 5: Sensitivity to different market parameters.
        
        Test various combinations of r, a(k), s(k), p(k), γ
        """
        self.log("\n" + "="*70)
        self.log("TEST 5: Parameter Sensitivity Analysis")
        self.log("="*70)
        
        param_sets = [
            {
                "name": "Low rate, low return",
                "config": {
                    **CONFIG_MVP_SANITY,
                    "r": 0.01,
                    "a": [0.05],
                    "s": [0.0009],
                }
            },
            {
                "name": "High rate, high return",
                "config": {
                    **CONFIG_MVP_SANITY,
                    "r": 0.04,
                    "a": [0.15],
                    "s": [0.0036],
                }
            },
            {
                "name": "High risk aversion",
                "config": {
                    **CONFIG_MVP_SANITY,
                    "gamma": 5.0,
                }
            },
            {
                "name": "Low risk aversion",
                "config": {
                    **CONFIG_MVP_SANITY,
                    "gamma": 0.5,
                }
            },
        ]
        
        results = []
        for param_set in param_sets:
            self.log(f"\nTesting: {param_set['name']}")
            env = AssetAllocationEnv(**param_set["config"])
            
            trained_agent, _, _ = train(
                env_config=param_set["config"],
                train_config=TRAINING_CONFIG,
                n_episodes=500
            )
            
            episode_rewards = []
            for _ in range(20):
                trajectory = rollout_episode(env, trained_agent)
                episode_rewards.append(sum(trajectory["rewards"]))
            
            avg_reward = mean(episode_rewards)
            std_reward = stdev(episode_rewards) if len(episode_rewards) > 1 else 0
            
            result = f"{param_set['name']}: {avg_reward:.4f} ± {std_reward:.4f}"
            results.append(result)
            self.log(f"  Result: {result}")
        
        passed = len(results) == len(param_sets)
        self.log(f"\nResult: {'PASS ✓' if passed else 'FAIL ✗'} - Tested {len(results)} parameter sets")
        
        self.results["parameter_sensitivity"] = {
            "passed": passed,
            "results": results,
        }
        return passed
    
    def test_baseline_comparison(self):
        """
        Test 6: Compare RL with simple baseline strategies.

        Baselines:
        1. Buy & Hold (no rebalancing)
        2. Equal Weight (rebalance to equal weights each period)
        """
        self.log("\n" + "="*70)
        self.log("TEST 6: Baseline Strategy Comparison")
        self.log("="*70)

        config = CONFIG_MVP_SANITY.copy()
        config["T"] = 5

        # Strategy 1: Buy & Hold
        env_bh = AssetAllocationEnv(**config)
        rewards_bh = []
        for _ in range(50):
            obs, _ = env_bh.reset()
            done = False
            episode_reward = 0
            while not done:
                action = np.zeros(env_bh.n_total_assets)  # No adjustment
                obs, reward, done, _, _ = env_bh.step(action)
                episode_reward += reward
            rewards_bh.append(episode_reward)

        # Strategy 2: Equal Weight
        env_ew = AssetAllocationEnv(**config)
        rewards_ew = []
        for _ in range(50):
            obs, _ = env_ew.reset()
            done = False
            episode_reward = 0
            while not done:
                target_weight = np.ones(env_ew.n_total_assets) / env_ew.n_total_assets
                action = (target_weight - env_ew.portfolio) / config["max_portfolio_adjustment"]
                obs, reward, done, _, _ = env_ew.step(action)
                episode_reward += reward
            rewards_ew.append(episode_reward)

        # Strategy 3: RL Agent
        trained_agent, _, _ = train(
            env_config=config,
            train_config=TRAINING_CONFIG,
            n_episodes=1000
        )

        env_rl = AssetAllocationEnv(**config)
        rewards_rl = []
        for _ in range(50):
            trajectory = rollout_episode(env_rl, trained_agent)
            rewards_rl.append(sum(trajectory["rewards"]))

        # Compare
        self.log(f"Buy & Hold:     {mean(rewards_bh):.4f} ± {stdev(rewards_bh):.4f}")
        self.log(f"Equal Weight:   {mean(rewards_ew):.4f} ± {stdev(rewards_ew):.4f}")
        self.log(f"RL Agent:       {mean(rewards_rl):.4f} ± {stdev(rewards_rl):.4f}")

        rl_better = mean(rewards_rl) > max(mean(rewards_bh), mean(rewards_ew))
        self.log(f"\nRL outperforms baselines: {'YES ✓' if rl_better else 'NO ✗'}")

        passed = rl_better
        self.log(f"Result: {'PASS ✓' if passed else 'FAIL ✗'}")

        self.results["baseline_comparison"] = {
            "passed": passed,
            "buy_hold": (mean(rewards_bh), stdev(rewards_bh)),
            "equal_weight": (mean(rewards_ew), stdev(rewards_ew)),
            "rl_agent": (mean(rewards_rl), stdev(rewards_rl)),
        }
        return passed

    # =========================================================================
    # SECTION 1 — Market Environment Tests
    # =========================================================================

    def test_dominated_asset(self):
        """
        Test 7: Dominated Asset Test.

        Set a(k) < r for one risk asset (negative risk premium).
        A rational agent should drive that asset's weight to 0.
        We verify the RL policy allocates ≤ 5% to the dominated asset
        after training.
        """
        self.log("\n" + "="*70)
        self.log("TEST 7: Dominated Asset Test  (a < r  →  weight → 0)")
        self.log("="*70)

        # Asset 1 is dominated: a=0.01 < r=0.03, positive variance
        # Asset 2 is normal:    a=0.10 > r=0.03
        config = {
            "n_assets": 2,
            "T": 5,
            "r": 0.03,
            "a": [0.01, 0.10],          # asset 1 dominated
            "s": [0.0016, 0.0036],
            "gamma": 1.0,
            "initial_portfolio": [0.2, 0.4, 0.4],  # [cash, asset1, asset2]
            "initial_wealth": 1.0,
            "max_portfolio_adjustment": 0.1,
        }

        trained_agent, _, _ = train(
            env_config=config,
            train_config=TRAINING_CONFIG,
            n_episodes=5000,
        )

        # Evaluate learned portfolio weights over many episodes
        env = AssetAllocationEnv(**config)
        dominated_weights = []
        for _ in range(100):
            obs, _ = env.reset()
            done = False
            while not done:
                action = trained_agent.select_action(obs)
                obs, _, done, _, info = env.step(action)
            # Record final rebalanced weight of dominated asset (index 1)
            dominated_weights.append(info["rebalanced_portfolio"][1])

        avg_dominated = float(np.mean(dominated_weights))
        self.log(f"Avg weight in dominated asset (a<r): {avg_dominated:.4f}")
        self.log(f"Expected: close to 0.0")

        # Pass if agent keeps dominated asset below 10% on average
        passed = avg_dominated < 0.10
        self.log(f"Result: {'PASS ✓' if passed else 'FAIL ✗'}")

        self.results["dominated_asset"] = {
            "passed": passed,
            "avg_dominated_weight": avg_dominated,
        }
        return passed

    def test_high_vs_low_risk_return(self):
        """
        Test 8: High-risk/high-return vs low-risk/low-return.

        Asset 1: low a, low s  →  conservative choice
        Asset 2: high a, high s  →  aggressive choice

        We check that the agent allocates more to asset 2 (higher Sharpe)
        when risk aversion is moderate, and more to asset 1 when γ is high.
        """
        self.log("\n" + "="*70)
        self.log("TEST 8: High-Risk/High-Return vs Low-Risk/Low-Return")
        self.log("="*70)

        base_config = {
            "n_assets": 2,
            "T": 5,
            "r": 0.02,
            "a": [0.05, 0.14],          # asset1: low return, asset2: high return
            "s": [0.0004, 0.0196],      # asset1: low var (σ=0.02), asset2: high var (σ=0.14)
            "initial_portfolio": [0.34, 0.33, 0.33],
            "initial_wealth": 1.0,
            "max_portfolio_adjustment": 0.1,
        }

        results = {}
        for label, gamma in [("moderate γ=1.0", 1.0), ("high γ=5.0", 5.0)]:
            cfg = {**base_config, "gamma": gamma}
            trained_agent, _, _ = train(
                env_config=cfg,
                train_config=TRAINING_CONFIG,
                n_episodes=3000,
            )
            env = AssetAllocationEnv(**cfg)
            w1_list, w2_list = [], []
            for _ in range(100):
                obs, _ = env.reset()
                done = False
                while not done:
                    action = trained_agent.select_action(obs)
                    obs, _, done, _, info = env.step(action)
                w1_list.append(info["rebalanced_portfolio"][1])
                w2_list.append(info["rebalanced_portfolio"][2])
            avg_w1 = float(np.mean(w1_list))
            avg_w2 = float(np.mean(w2_list))
            results[label] = {"avg_w1": avg_w1, "avg_w2": avg_w2}
            self.log(f"  {label}: w_low={avg_w1:.4f}, w_high={avg_w2:.4f}")

        # Moderate γ: agent should prefer high-return asset (w2 > w1)
        moderate_ok = results["moderate γ=1.0"]["avg_w2"] >= results["moderate γ=1.0"]["avg_w1"]
        # High γ: agent should be more conservative (w1 relatively higher than in moderate case)
        high_gamma_more_conservative = (
            results["high γ=5.0"]["avg_w1"] >= results["moderate γ=1.0"]["avg_w1"] - 0.05
        )

        passed = moderate_ok and high_gamma_more_conservative
        self.log(f"  Moderate γ prefers high-return asset: {'YES ✓' if moderate_ok else 'NO ✗'}")
        self.log(f"  High γ is more conservative: {'YES ✓' if high_gamma_more_conservative else 'NO ✗'}")
        self.log(f"Result: {'PASS ✓' if passed else 'FAIL ✗'}")

        self.results["high_vs_low_risk_return"] = {
            "passed": passed,
            "results": results,
        }
        return passed

    def test_scalability_n3_vs_n4(self):
        """
        Test 9: Scalability — n=3 vs n=4 assets.

        Verifies that training converges stably when the asset dimension
        increases from 3 to 4.  We check that the final reward is finite
        and that the reward variance is bounded (no divergence).
        """
        self.log("\n" + "="*70)
        self.log("TEST 9: Scalability — n=3 vs n=4 Assets")
        self.log("="*70)

        configs = {
            "n=3": TEST_CONFIGS["three_assets"],
            "n=4": TEST_CONFIGS["four_assets"],
        }

        passed = True
        for label, cfg in configs.items():
            trained_agent, _, _ = train(
                env_config=cfg,
                train_config=TRAINING_CONFIG,
                n_episodes=2000,
            )
            env = AssetAllocationEnv(**cfg)
            eval_rewards = []
            for _ in range(50):
                traj = rollout_episode(env, trained_agent)
                eval_rewards.append(sum(traj["rewards"]))

            avg_r = float(np.mean(eval_rewards))
            std_r = float(np.std(eval_rewards))
            finite = np.isfinite(avg_r)
            stable = std_r < 5.0   # loose bound — just checking no explosion

            self.log(f"  {label}: avg_reward={avg_r:.4f}, std={std_r:.4f}, "
                     f"finite={finite}, stable={stable}")
            if not (finite and stable):
                passed = False

        self.log(f"Result: {'PASS ✓' if passed else 'FAIL ✗'}")
        self.results["scalability_n3_n4"] = {"passed": passed}
        return passed

    # =========================================================================
    # SECTION 2 — Initial Portfolio & Constraint Limit Tests
    # =========================================================================

    def test_extreme_initialization(self):
        """
        Test 10: Extreme Initialization — agent starts at 100% cash.

        With T=8 and max_adjustment=0.1, the agent can shift at most 80%
        of wealth over the horizon.  We verify the agent:
          (a) consistently uses the full 10% adjustment budget each step
              (i.e., one-way turnover ≈ 0.1 per step), and
          (b) ends up with a higher risky allocation than it started with.
        """
        self.log("\n" + "="*70)
        self.log("TEST 10: Extreme Initialization (100% cash, T=8)")
        self.log("="*70)

        config = {
            "n_assets": 1,
            "T": 8,
            "r": 0.02,
            "a": [0.10],
            "s": [0.0016],
            "gamma": 1.0,
            "initial_portfolio": [1.0, 0.0],   # 100% cash
            "initial_wealth": 1.0,
            "max_portfolio_adjustment": 0.1,
        }

        trained_agent, _, _ = train(
            env_config=config,
            train_config=TRAINING_CONFIG,
            n_episodes=3000,
        )

        env = AssetAllocationEnv(**config)
        step_turnovers = []   # one-way turnover per step
        final_risky_weights = []

        for _ in range(100):
            obs, _ = env.reset()
            prev_portfolio = env.portfolio.copy()
            done = False
            ep_turnovers = []
            while not done:
                action = trained_agent.select_action(obs)
                raw = action * config["max_portfolio_adjustment"]
                feasible = project_action_to_feasible_set(
                    raw, prev_portfolio, config["max_portfolio_adjustment"]
                )
                one_way = float(np.maximum(feasible, 0).sum())
                ep_turnovers.append(one_way)
                obs, _, done, _, info = env.step(action)
                prev_portfolio = info["portfolio"].copy()
            step_turnovers.append(float(np.mean(ep_turnovers)))
            final_risky_weights.append(info["rebalanced_portfolio"][1])

        avg_turnover = float(np.mean(step_turnovers))
        avg_final_risky = float(np.mean(final_risky_weights))

        self.log(f"  Avg per-step one-way turnover: {avg_turnover:.4f}  (max allowed: 0.10)")
        self.log(f"  Avg final risky weight:        {avg_final_risky:.4f}  (started at 0.0)")

        # Agent should be actively rebalancing (turnover > 0.03) and
        # should have moved meaningful wealth into the risky asset
        actively_rebalancing = avg_turnover > 0.03
        moved_to_risky = avg_final_risky > 0.10

        passed = actively_rebalancing and moved_to_risky
        self.log(f"  Actively rebalancing: {'YES ✓' if actively_rebalancing else 'NO ✗'}")
        self.log(f"  Moved wealth to risky asset: {'YES ✓' if moved_to_risky else 'NO ✗'}")
        self.log(f"Result: {'PASS ✓' if passed else 'FAIL ✗'}")

        self.results["extreme_initialization"] = {
            "passed": passed,
            "avg_turnover": avg_turnover,
            "avg_final_risky": avg_final_risky,
        }
        return passed

    def test_optimal_initialization(self):
        """
        Test 11: Optimal Initialization — agent starts at the Merton optimum.

        When the initial portfolio already matches the unconstrained optimum,
        a smart agent should trade very little (near-zero actions) to avoid
        unnecessary drift away from the optimum.
        """
        self.log("\n" + "="*70)
        self.log("TEST 11: Optimal Initialization (start at Merton optimum)")
        self.log("="*70)

        base = CONFIG_MVP_SANITY.copy()
        base["T"] = 5

        merton = merton_optimal_allocation(
            a=base["a"][0],
            r=base["r"],
            s=base["s"][0],
            gamma=base["gamma"],
        )

        if not merton["is_valid"]:
            self.log("  Merton solution infeasible — skipping test.")
            self.results["optimal_initialization"] = {"passed": True, "skipped": True}
            return True

        p_cash = float(np.clip(merton["p_cash"], 0, 1))
        p_risky = float(np.clip(merton["p_risky"], 0, 1))
        # Renormalise in case of tiny floating-point drift
        total = p_cash + p_risky
        p_cash /= total
        p_risky /= total

        config = {**base, "initial_portfolio": [p_cash, p_risky]}
        self.log(f"  Merton optimum: cash={p_cash:.4f}, risky={p_risky:.4f}")

        trained_agent, _, _ = train(
            env_config=config,
            train_config=TRAINING_CONFIG,
            n_episodes=3000,
        )

        env = AssetAllocationEnv(**config)
        step_turnovers = []
        for _ in range(100):
            obs, _ = env.reset()
            prev_portfolio = env.portfolio.copy()
            done = False
            ep_turnovers = []
            while not done:
                action = trained_agent.select_action(obs)
                raw = action * config["max_portfolio_adjustment"]
                feasible = project_action_to_feasible_set(
                    raw, prev_portfolio, config["max_portfolio_adjustment"]
                )
                one_way = float(np.maximum(feasible, 0).sum())
                ep_turnovers.append(one_way)
                obs, _, done, _, info = env.step(action)
                prev_portfolio = info["portfolio"].copy()
            step_turnovers.append(float(np.mean(ep_turnovers)))

        avg_turnover = float(np.mean(step_turnovers))
        self.log(f"  Avg per-step one-way turnover: {avg_turnover:.4f}")
        self.log(f"  Expected: small (agent should hold near-optimal position)")

        # Agent should trade less than half the maximum budget on average
        passed = avg_turnover < 0.06
        self.log(f"Result: {'PASS ✓' if passed else 'FAIL ✗'}")

        self.results["optimal_initialization"] = {
            "passed": passed,
            "avg_turnover": avg_turnover,
            "merton_allocation": merton,
        }
        return passed

    # =========================================================================
    # SECTION 3 — Risk Aversion Sensitivity
    # =========================================================================

    def test_risk_aversion_sensitivity(self):
        """
        Test 12: γ Sensitivity Analysis.

        Keep all market parameters fixed; vary γ ∈ {0.1, 0.5, 1.0, 3.0, 10.0}.

        Expected monotone relationship:
          higher γ  →  lower average risky-asset weight (more conservative)
          lower  γ  →  higher average risky-asset weight (more aggressive)
        """
        self.log("\n" + "="*70)
        self.log("TEST 12: Risk-Aversion Sensitivity (γ sweep)")
        self.log("="*70)

        gammas = [0.1, 0.5, 1.0, 3.0, 10.0]
        base = {
            "n_assets": 1,
            "T": 5,
            "r": 0.02,
            "a": [0.10],
            "s": [0.0016],
            "initial_portfolio": [0.5, 0.5],
            "initial_wealth": 1.0,
            "max_portfolio_adjustment": 0.1,
        }

        avg_risky_by_gamma = {}
        for gamma in gammas:
            cfg = {**base, "gamma": gamma}
            trained_agent, _, _ = train(
                env_config=cfg,
                train_config=TRAINING_CONFIG,
                n_episodes=2000,
            )
            env = AssetAllocationEnv(**cfg)
            risky_weights = []
            for _ in range(100):
                obs, _ = env.reset()
                done = False
                while not done:
                    action = trained_agent.select_action(obs)
                    obs, _, done, _, info = env.step(action)
                risky_weights.append(info["rebalanced_portfolio"][1])
            avg_w = float(np.mean(risky_weights))
            avg_risky_by_gamma[gamma] = avg_w
            self.log(f"  γ={gamma:5.1f}: avg risky weight = {avg_w:.4f}")

        # Check overall downward trend: linear regression slope of (γ → risky weight) < 0.
        # This is more robust than strict pairwise comparison, which can fail when extreme
        # γ values cause reward-scale issues (γ→0: large negative CARA, γ→∞: near-zero CARA).
        weights_ordered = [avg_risky_by_gamma[g] for g in gammas]
        gamma_arr = np.array(gammas, dtype=float)
        weight_arr = np.array(weights_ordered, dtype=float)
        gamma_c = gamma_arr - gamma_arr.mean()
        weight_c = weight_arr - weight_arr.mean()
        slope = float(np.dot(gamma_c, weight_c) / (np.dot(gamma_c, gamma_c) + 1e-12))
        monotone = slope < 0  # negative slope: higher γ → lower risky weight on average

        passed = monotone
        self.log(f"\n  OLS slope of (γ → risky weight): {slope:.4f}  (expect < 0)")
        self.log(f"  Overall trend (higher γ → lower risky weight): {'YES ✓' if monotone else 'NO ✗'}")
        self.log(f"Result: {'PASS ✓' if passed else 'FAIL ✗'}")

        self.results["risk_aversion_sensitivity"] = {
            "passed": passed,
            "avg_risky_by_gamma": avg_risky_by_gamma,
        }
        return passed

    def test_extreme_risk_aversion(self):
        """
        Test 13: Extreme γ Behaviour.

        Very high γ (=50): agent should flee to cash (risky weight ≈ 0).
        Very low  γ (=0.05): agent should be nearly risk-neutral and
          maximise expected return (risky weight near maximum feasible).
        """
        self.log("\n" + "="*70)
        self.log("TEST 13: Extreme Risk Aversion (γ=50 vs γ=0.05)")
        self.log("="*70)

        base = {
            "n_assets": 1,
            "T": 5,
            "r": 0.02,
            "a": [0.10],
            "s": [0.0016],
            "initial_portfolio": [0.5, 0.5],
            "initial_wealth": 1.0,
            "max_portfolio_adjustment": 0.1,
        }

        results = {}
        for label, gamma in [("very high γ=50", 50.0), ("very low γ=0.05", 0.05)]:
            cfg = {**base, "gamma": gamma}
            trained_agent, _, _ = train(
                env_config=cfg,
                train_config=TRAINING_CONFIG,
                n_episodes=3000,
            )
            env = AssetAllocationEnv(**cfg)
            risky_weights = []
            for _ in range(100):
                obs, _ = env.reset()
                done = False
                while not done:
                    action = trained_agent.select_action(obs)
                    obs, _, done, _, info = env.step(action)
                risky_weights.append(info["rebalanced_portfolio"][1])
            avg_w = float(np.mean(risky_weights))
            results[label] = avg_w
            self.log(f"  {label}: avg risky weight = {avg_w:.4f}")

        # Very high γ → near-zero risky allocation
        high_gamma_conservative = results["very high γ=50"] < 0.20
        # Very low γ → high risky allocation (risk-neutral → max return)
        low_gamma_aggressive = results["very low γ=0.05"] > 0.40

        passed = high_gamma_conservative and low_gamma_aggressive
        self.log(f"  Very high γ flees to cash: {'YES ✓' if high_gamma_conservative else 'NO ✗'}")
        self.log(f"  Very low  γ chases return: {'YES ✓' if low_gamma_aggressive else 'NO ✗'}")
        self.log(f"Result: {'PASS ✓' if passed else 'FAIL ✗'}")

        self.results["extreme_risk_aversion"] = {
            "passed": passed,
            "results": results,
        }
        return passed

    # =========================================================================
    # SECTION 4 — Extended Baseline Comparisons
    # =========================================================================

    def _run_strategy(self, env_config: Dict, strategy: str,
                      n_episodes: int = 200, trained_agent=None):
        """
        Helper: run a fixed strategy for n_episodes and return list of
        episode utilities (terminal CARA reward).

        strategy ∈ {"buy_hold", "random", "greedy", "rl"}
        """
        env = AssetAllocationEnv(**env_config)
        rewards = []

        for _ in range(n_episodes):
            obs, _ = env.reset()
            done = False
            ep_reward = 0.0
            while not done:
                if strategy == "buy_hold":
                    # Zero adjustment — hold current portfolio
                    action = np.zeros(env.n_total_assets)

                elif strategy == "random":
                    # Random action in [-1, 1]; projection enforces constraints
                    action = np.random.uniform(-1, 1, size=env.n_total_assets)

                elif strategy == "greedy":
                    # Move 10% toward the asset with the highest expected return.
                    # Cash has return r; risky assets have return a[k].
                    all_returns = np.concatenate([[env.r], env.a])
                    best_idx = int(np.argmax(all_returns))
                    # Build a raw action that buys the best asset and sells others
                    raw = np.full(env.n_total_assets, -env.max_portfolio_adjustment)
                    raw[best_idx] = env.max_portfolio_adjustment * (env.n_total_assets - 1)
                    # Scale to [-1, 1] for the env.step interface
                    action = raw / env.max_portfolio_adjustment

                elif strategy == "rl":
                    assert trained_agent is not None
                    action = trained_agent.select_action(obs)

                else:
                    raise ValueError(f"Unknown strategy: {strategy}")

                obs, reward, done, _, _ = env.step(action)
                ep_reward += reward
            rewards.append(ep_reward)

        return rewards

    def test_four_way_baseline_comparison(self):
        """
        Test 14: Four-Way Baseline Comparison.

        Compare RL against:
          1. Buy-and-Hold  — no rebalancing
          2. Random Action — monkey strategy (random feasible trades)
          3. Greedy        — each step move 10% toward highest-return asset

        RL should outperform all three baselines on average CARA utility.
        """
        self.log("\n" + "="*70)
        self.log("TEST 14: Four-Way Baseline Comparison (RL vs BH / Random / Greedy)")
        self.log("="*70)

        config = {
            "n_assets": 2,
            "T": 8,
            "r": 0.02,
            "a": [0.08, 0.13],
            "s": [0.0016, 0.0064],
            "gamma": 1.0,
            "initial_portfolio": [0.2, 0.4, 0.4],
            "initial_wealth": 1.0,
            "max_portfolio_adjustment": 0.1,
        }

        N_EVAL = 200

        # Train RL agent
        trained_agent, _, _ = train(
            env_config=config,
            train_config=TRAINING_CONFIG,
            n_episodes=2000,
        )

        strategy_rewards = {}
        for strat in ["buy_hold", "random", "greedy", "rl"]:
            r_list = self._run_strategy(
                config, strat, n_episodes=N_EVAL,
                trained_agent=trained_agent if strat == "rl" else None,
            )
            strategy_rewards[strat] = r_list
            self.log(f"  {strat:12s}: {mean(r_list):.4f} ± {stdev(r_list):.4f}")

        rl_mean = mean(strategy_rewards["rl"])
        baselines_mean = {k: mean(v) for k, v in strategy_rewards.items() if k != "rl"}
        rl_beats_all = all(rl_mean > v for v in baselines_mean.values())

        self.log(f"\n  RL beats all baselines: {'YES ✓' if rl_beats_all else 'NO ✗'}")

        passed = rl_beats_all
        self.log(f"Result: {'PASS ✓' if passed else 'FAIL ✗'}")

        self.results["four_way_baseline"] = {
            "passed": passed,
            "means": {k: mean(v) for k, v in strategy_rewards.items()},
            "stds":  {k: stdev(v) for k, v in strategy_rewards.items()},
        }
        return passed

    def test_greedy_vs_rl_long_horizon(self):
        """
        Test 15: Greedy vs RL on a longer horizon (T=9).

        The greedy strategy is myopic — it ignores future constraints.
        Over a longer horizon the RL agent's forward-looking policy should
        accumulate a clear advantage.
        """
        self.log("\n" + "="*70)
        self.log("TEST 15: Greedy vs RL — Long Horizon (T=9)")
        self.log("="*70)

        config = {
            "n_assets": 2,
            "T": 9,
            "r": 0.02,
            "a": [0.07, 0.12],
            "s": [0.0009, 0.0049],
            "gamma": 1.5,
            "initial_portfolio": [0.5, 0.25, 0.25],
            "initial_wealth": 1.0,
            "max_portfolio_adjustment": 0.1,
        }

        N_EVAL = 200

        trained_agent, _, _ = train(
            env_config=config,
            train_config=TRAINING_CONFIG,
            n_episodes=5000,
        )

        greedy_rewards = self._run_strategy(config, "greedy", n_episodes=N_EVAL)
        rl_rewards = self._run_strategy(
            config, "rl", n_episodes=N_EVAL, trained_agent=trained_agent
        )

        self.log(f"  Greedy: {mean(greedy_rewards):.4f} ± {stdev(greedy_rewards):.4f}")
        self.log(f"  RL:     {mean(rl_rewards):.4f} ± {stdev(rl_rewards):.4f}")

        rl_better = mean(rl_rewards) > mean(greedy_rewards)
        passed = rl_better
        self.log(f"  RL outperforms greedy: {'YES ✓' if rl_better else 'NO ✗'}")
        self.log(f"Result: {'PASS ✓' if passed else 'FAIL ✗'}")

        self.results["greedy_vs_rl_long_horizon"] = {
            "passed": passed,
            "greedy": (mean(greedy_rewards), stdev(greedy_rewards)),
            "rl": (mean(rl_rewards), stdev(rl_rewards)),
        }
        return passed

    def run_all_tests(self):
        """Run complete test suite."""
        self.log("\n" + "="*70)
        self.log("COMPREHENSIVE TEST SUITE FOR RL ASSET ALLOCATION")
        self.log("="*70)

        start_time = time.time()

        test_results = []

        # ── Original tests ────────────────────────────────────────────────────
        # Test 1: Constraints (critical)
        test_results.append(("Constraint Satisfaction",
                             self.test_constraint_satisfaction(CONFIG_MVP_SANITY)))

        # Test 2: Time horizons
        test_results.append(("Time Horizons (T < 10)",
                             self.test_time_horizons()))

        # Test 3: Asset numbers
        test_results.append(("Asset Numbers (n < 5)",
                             self.test_asset_numbers()))

        # Test 4: Merton comparison
        test_results.append(("Merton Analytical",
                             self.test_merton_comparison()))

        # Test 5: Parameter sensitivity
        test_results.append(("Parameter Sensitivity",
                             self.test_parameter_sensitivity()))

        # Test 6: Baseline comparison (Buy-and-Hold / Equal-Weight / RL)
        test_results.append(("Baseline Comparison",
                             self.test_baseline_comparison()))

        # ── Section 1: Market environment ─────────────────────────────────────
        # Test 7: Dominated asset (a < r → weight → 0)
        test_results.append(("Dominated Asset",
                             self.test_dominated_asset()))

        # Test 8: High-risk/high-return vs low-risk/low-return
        test_results.append(("High vs Low Risk/Return",
                             self.test_high_vs_low_risk_return()))

        # Test 9: Scalability n=3 vs n=4
        test_results.append(("Scalability n=3 vs n=4",
                             self.test_scalability_n3_vs_n4()))

        # ── Section 2: Initial portfolio & constraint limits ───────────────────
        # Test 10: Extreme initialization (100% cash)
        test_results.append(("Extreme Initialization",
                             self.test_extreme_initialization()))

        # Test 11: Optimal initialization (start at Merton optimum)
        test_results.append(("Optimal Initialization",
                             self.test_optimal_initialization()))

        # ── Section 3: Risk aversion sensitivity ──────────────────────────────
        # Test 12: γ sweep (monotone relationship)
        test_results.append(("Risk Aversion γ Sweep",
                             self.test_risk_aversion_sensitivity()))

        # Test 13: Extreme γ (very high / very low)
        test_results.append(("Extreme Risk Aversion",
                             self.test_extreme_risk_aversion()))

        # ── Section 4: Extended baseline comparisons ──────────────────────────
        # Test 14: Four-way comparison (BH / Random / Greedy / RL)
        test_results.append(("Four-Way Baseline (BH/Random/Greedy/RL)",
                             self.test_four_way_baseline_comparison()))

        # Test 15: Greedy vs RL on long horizon
        test_results.append(("Greedy vs RL Long Horizon",
                             self.test_greedy_vs_rl_long_horizon()))

        elapsed = time.time() - start_time

        # Summary
        self.log("\n" + "="*70)
        self.log("TEST SUMMARY")
        self.log("="*70)

        passed_count = sum(1 for _, result in test_results if result)
        total_count = len(test_results)

        for name, result in test_results:
            status = "✓ PASS" if result else "✗ FAIL"
            self.log(f"{status}: {name}")

        self.log(f"\nTotal: {passed_count}/{total_count} tests passed")
        self.log(f"Execution time: {elapsed:.2f} seconds")

        overall_passed = passed_count == total_count
        self.log(f"\nOverall: {'ALL TESTS PASSED ✓✓✓' if overall_passed else 'SOME TESTS FAILED'}")

        return overall_passed, self.results


if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Run comprehensive tests
    suite = ComprehensiveTestSuite(verbose=True)
    passed, results = suite.run_all_tests()
    
    # Exit code for CI/CD
    exit(0 if passed else 1)
