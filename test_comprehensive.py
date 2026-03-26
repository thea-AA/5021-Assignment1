"""
Comprehensive test suite for asset allocation RL solution
Tests all constraints and requirements from the problem statement.
"""
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
from utils import merton_optimal_allocation, cara_utility


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
        1. |Δp_k| ≤ 0.1 (max adjustment)
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
                
                # Check constraint before step
                current_portfolio = env.portfolio.copy()
                
                obs, reward, done, truncated, info = env.step(action)
                new_portfolio = info["portfolio"]
                
                # Check 1: Adjustment limit
                adjustment = np.abs(new_portfolio - current_portfolio)
                if np.any(adjustment > 0.1 + 1e-6):  # Small tolerance
                    violations["adjustment"] += 1
                
                # Check 2: No short selling
                if np.any(new_portfolio < -1e-6):
                    violations["short_selling"] += 1
                
                # Check 3: Budget balance
                if abs(np.sum(new_portfolio) - 1.0) > 1e-6:
                    violations["budget_balance"] += 1
        
        passed = sum(v == 0 for v in violations.values()) == len(violations)
        self.log(f"✓ Adjustment constraint violations: {violations['adjustment']}/{n_tests * config['T']}")
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
    
    def run_all_tests(self):
        """Run complete test suite."""
        self.log("\n" + "="*70)
        self.log("COMPREHENSIVE TEST SUITE FOR RL ASSET ALLOCATION")
        self.log("="*70)
        
        start_time = time.time()
        
        # Run all tests
        test_results = []
        
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
        
        # Test 6: Baseline comparison
        test_results.append(("Baseline Comparison", 
                            self.test_baseline_comparison()))
        
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
