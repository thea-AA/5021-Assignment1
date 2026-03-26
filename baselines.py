"""
Baseline strategies for comparison with RL agent.
"""
import numpy as np
from typing import Dict, List


class BaselineStrategies:
    """Collection of baseline portfolio strategies."""
    
    @staticmethod
    def buy_and_hold(env):
        """
        Buy & Hold: Initial allocation, no rebalancing.
        
        Action: Zero adjustment at all periods.
        """
        return np.zeros(env.n_total_assets)
    
    @staticmethod
    def equal_weight(env):
        """
        Equal Weight: Rebalance to equal weights each period.
        
        Action: Move towards 1/n allocation for all assets.
        """
        target = np.ones(env.n_total_assets) / env.n_total_assets
        adjustment = target - env.portfolio
        
        # Scale to action space [-1, 1]
        action = adjustment / env.max_portfolio_adjustment
        return action
    
    @staticmethod
    def merton_static(env, a, r, s, gamma):
        """
        Static Merton Allocation: Compute optimal once, try to maintain.
        
        For multi-asset case, use independent Merton for each asset
        (suboptimal but simple baseline).
        
        Args:
            a: Expected returns (array)
            r: Risk-free rate
            s: Variances (array)
            gamma: Risk aversion
        """
        n_risky = len(a)
        target = np.zeros(env.n_total_assets)
        
        # Compute Merton allocation for each risky asset
        for i in range(n_risky):
            numerator = a[i] - r
            denominator = gamma * s[i]
            
            if abs(denominator) > 1e-10:
                target[i + 1] = numerator / denominator  # +1 for cash index
            else:
                target[i + 1] = 0.0
        
        # Ensure no short selling
        target = np.maximum(target, 0.0)
        
        # Normalize to sum to 1
        total = np.sum(target)
        if total > 0:
            target = target / total
        else:
            target = np.ones(env.n_total_assets) / env.n_total_assets
        
        # Adjust cash
        target[0] = 1.0 - np.sum(target[1:])
        
        # Compute action to move towards target
        adjustment = target - env.portfolio
        action = adjustment / env.max_portfolio_adjustment
        return action
    
    @staticmethod
    def max_sharpe(env, a, r, s):
        """
        Maximum Sharpe Ratio: Tangency portfolio.
        
        For single risk asset:
        p* proportional to (a - r) / s
        
        Args:
            a: Expected returns
            r: Risk-free rate
            s: Variances
        """
        n_risky = len(a)
        weights = np.zeros(n_risky)
        
        for i in range(n_risky):
            excess_return = a[i] - r
            if s[i] > 1e-10:
                weights[i] = excess_return / s[i]
            else:
                weights[i] = 0.0
        
        # Ensure non-negative
        weights = np.maximum(weights, 0.0)
        
        # Normalize risky weights
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)
        else:
            weights = np.ones(n_risky) / n_risky
        
        # Construct full portfolio [cash, risky1, risky2, ...]
        target = np.zeros(env.n_total_assets)
        target[1:] = weights
        target[0] = 0.0  # All in risky assets (can be modified)
        
        # Adjust to current portfolio
        adjustment = target - env.portfolio
        action = adjustment / env.max_portfolio_adjustment
        return action
    
    @staticmethod
    def momentum(env, past_returns: np.ndarray, lookback: int = 3):
        """
        Momentum Strategy: Increase allocation to assets with high past returns.
        
        Args:
            past_returns: Array of shape (T_lookback, n_risky)
            lookback: Number of periods to look back
        """
        if len(past_returns) < lookback:
            # Not enough history, use equal weight
            return BaselineStrategies.equal_weight(env)
        
        # Compute average past returns
        avg_returns = np.mean(past_returns[-lookback:], axis=0)
        
        # Simple momentum: overweight high return assets
        scores = np.softmax(avg_returns * 10)  # Scale for numerical stability
        
        target = np.zeros(env.n_total_assets)
        target[1:] = scores
        target[0] = 0.2  # Keep 20% in cash
        
        adjustment = target - env.portfolio
        action = adjustment / env.max_portfolio_adjustment
        return action
    
    @staticmethod
    def min_variance(env, cov_matrix: np.ndarray):
        """
        Minimum Variance Portfolio.
        
        Args:
            cov_matrix: Covariance matrix of asset returns (n_risky x n_risky)
        """
        n_risky = cov_matrix.shape[0]
        
        try:
            # Min variance weights: w = Σ^(-1) 1 / (1' Σ^(-1) 1)
            inv_cov = np.linalg.inv(cov_matrix)
            ones = np.ones(n_risky)
            w = inv_cov @ ones
            w = w / np.sum(w)
            w = np.maximum(w, 0.0)  # No short selling
            w = w / np.sum(w)  # Renormalize
        except:
            # Fallback to equal weight
            w = np.ones(n_risky) / n_risky
        
        target = np.zeros(env.n_total_assets)
        target[1:] = w
        target[0] = 0.3  # Keep 30% in cash
        
        adjustment = target - env.portfolio
        action = adjustment / env.max_portfolio_adjustment
        return action


def evaluate_strategy(env, strategy_fn, n_episodes: int = 100, **kwargs):
    """
    Evaluate a strategy over multiple episodes.
    
    Args:
        env: Gym environment
        strategy_fn: Function that maps env -> action
        n_episodes: Number of evaluation episodes
        **kwargs: Additional arguments passed to strategy_fn
    
    Returns:
        Dict with mean reward, std, mean wealth, etc.
    """
    rewards = []
    final_wealths = []
    
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action = strategy_fn(env, **kwargs)
            obs, reward, done, _, info = env.step(action)
            episode_reward += reward
        
        rewards.append(episode_reward)
        final_wealths.append(info["wealth"])
    
    return {
        "mean_reward": np.mean(rewards),
        "std_reward": np.std(rewards),
        "mean_wealth": np.mean(final_wealths),
        "std_wealth": np.std(final_wealths),
        "min_wealth": np.min(final_wealths),
        "max_wealth": np.max(final_wealths),
    }
