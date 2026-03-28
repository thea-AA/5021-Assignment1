"""
Gym-style environment for discrete-time asset allocation
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from utils import project_action_to_feasible_set, reward_function


class AssetAllocationEnv(gym.Env):
    """
    Discrete-time portfolio allocation environment.

    State: [t/T, W/W0, p0, p1, ..., pN]  (normalized time, wealth, and portfolio)
    Action: [Δp0, Δp1, ..., ΔpN]  (portfolio adjustments, unconstrained)
    Reward: Only at terminal step = CARA utility of final wealth
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        n_assets,
        T,
        r,
        a,
        s,
        gamma,
        initial_portfolio,
        initial_wealth=1.0,
        max_portfolio_adjustment=0.1,
        cov_matrix=None,
        seed=None,
    ):
        """
        Args:
            n_assets: Number of risk assets (excluding cash)
            T: Time horizon (number of steps)
            r: Risk-free rate
            a: List of expected returns for each risk asset
            s: List of variances for each risk asset (NOT std dev)
            gamma: Absolute risk aversion coefficient
            initial_portfolio: List [p_cash, p_asset1, ..., p_assetN]
            initial_wealth: Initial wealth
            max_portfolio_adjustment: Max |Δp_k| per step
            cov_matrix: Optional covariance matrix (n_assets × n_assets).
                        If None, defaults to diag(s) — independent assets.
            seed: Random seed
        """
        super().__init__()

        self.n_assets = n_assets
        self.n_total_assets = n_assets + 1  # Including cash
        self.T = T
        self.r = r
        self.a = np.array(a)
        self.s = np.array(s)
        self.gamma = gamma
        self.initial_portfolio = np.array(initial_portfolio)
        self.initial_wealth = initial_wealth
        self.max_portfolio_adjustment = max_portfolio_adjustment

        # Covariance matrix: default to diag(s) for independent assets
        if cov_matrix is not None:
            self.cov_matrix = np.array(cov_matrix)
        else:
            self.cov_matrix = np.diag(self.s)  # diag(variance) = independent

        # State: [t/T, W/W0, p0, p1, ..., pN]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(2 + self.n_total_assets,),
            dtype=np.float32,
        )

        # Action: adjustments to portfolio (unconstrained, will be clipped)
        self.action_space = spaces.Box(
            low=-1.0,  # Will be scaled by max_portfolio_adjustment
            high=1.0,
            shape=(self.n_total_assets,),
            dtype=np.float32,
        )

        self.rng = np.random.RandomState(seed)

        # Internal state
        self.t = 0
        self.wealth = initial_wealth
        self.portfolio = self.initial_portfolio.copy()

    def reset(self, seed=None):
        """Reset environment to initial state."""
        if seed is not None:
            self.rng = np.random.RandomState(seed)

        self.t = 0
        self.wealth = self.initial_wealth
        self.portfolio = self.initial_portfolio.copy()

        return self._get_observation(), {}

    def _get_observation(self):
        """Return normalized observation."""
        obs = np.concatenate(
            [
                [self.t / max(1, self.T - 1)],  # Normalized time
                [self.wealth / self.initial_wealth],  # Normalized wealth
                self.portfolio,  # Portfolio allocation
            ]
        )
        return obs.astype(np.float32)

    def step(self, action):
        """
        Execute one step in the environment.

        Args:
            action: Raw action from RL agent (will be scaled and projected)

        Returns:
            observation, reward, terminated, truncated, info
        """
        if self.t >= self.T:
            raise RuntimeError("Episode already finished, call reset().")

        # Scale action from [-1, 1] to raw adjustment space, then project
        raw_action = action * self.max_portfolio_adjustment

        # Project action to feasible set (constraints A/B/C enforced inside)
        # After projection: Σ Δp = 0, so new_portfolio already sums to 1.
        feasible_action = project_action_to_feasible_set(
            raw_action, self.portfolio, self.max_portfolio_adjustment
        )
        rebalanced_portfolio = self.portfolio + feasible_action

        # Sample asset returns: R_k ~ N(a_k, Σ)
        asset_returns = self.rng.multivariate_normal(self.a, self.cov_matrix)

        # Portfolio return: r_p = p_cash * r + Σ p_k * R_k
        portfolio_return = (
            rebalanced_portfolio[0] * self.r
            + np.dot(rebalanced_portfolio[1:], asset_returns)
        )

        # New wealth
        new_wealth = self.wealth * (1.0 + portfolio_return)

        # ── Market Drift ──────────────────────────────────────────────────────
        # After earning returns, each asset's value grows at a different rate,
        # so weights naturally drift.  The state at the START of the next period
        # is the post-drift portfolio, NOT the pre-return rebalanced portfolio.
        #
        #   p_k^{t+1} = p_k^t * (1 + R_k) / Σ_j p_j^t * (1 + R_j)
        #
        # This is the standard discrete-time portfolio dynamics.
        gross = np.concatenate([[1.0 + self.r], 1.0 + asset_returns])
        drifted = rebalanced_portfolio * gross
        self.portfolio = drifted / drifted.sum()   # renormalise to simplex
        # ─────────────────────────────────────────────────────────────────────

        self.t += 1
        self.wealth = new_wealth

        reward = 0.0
        terminated = self.t >= self.T

        # Reward only at terminal step
        if terminated:
            reward = reward_function(self.wealth, self.gamma)

        obs = self._get_observation()
        info = {
            "portfolio": self.portfolio.copy(),   # post-drift weights (start of next period)
            "rebalanced_portfolio": rebalanced_portfolio.copy(),  # pre-drift (what was traded)
            "wealth": new_wealth,
            "asset_returns": asset_returns,
            "portfolio_return": portfolio_return,
        }

        return obs, float(reward), terminated, False, info

    def render(self):
        """Render environment state (not implemented)."""
        pass
