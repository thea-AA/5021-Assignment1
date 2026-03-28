"""
Utility functions and analytical solutions for verification
"""
import numpy as np


def cara_utility(wealth, gamma):
    """
    Absolute Risk Averse (CARA) utility function.
    U(W) = -exp(-gamma * W) / gamma
    For numerical stability: if gamma is small, approximate with linear term.
    """
    if gamma < 1e-6:
        return wealth
    try:
        exponent = np.clip(-gamma * wealth, -100, 100)
        return -np.exp(exponent) / gamma
    except (OverflowError, FloatingPointError, ValueError):
        return -np.inf


def merton_optimal_allocation(a, r, s, gamma):
    """
    Analytical optimal allocation for single risk asset + cash (unconstrained).
    From Merton portfolio problem / Rao & Jelvis 8.4.

    p* = (a - r) / (gamma * s)

    NOTE: s here is VARIANCE (not std dev)!

    Args:
        a: Expected return of risk asset
        r: Risk-free rate
        s: Variance of risk asset return (NOT std dev)
        gamma: Absolute risk aversion coefficient

    Returns:
        Optimal proportion in risk asset (remainder in cash)
    """
    numerator = a - r
    denominator = gamma * s  # s is already variance

    if abs(denominator) < 1e-10:
        return {
            "p_risky": float('inf') if numerator > 0 else float('-inf'),
            "p_cash": float('-inf'),
            "is_valid": False,
        }

    p_risky = numerator / denominator
    p_cash = 1.0 - p_risky

    return {
        "p_risky": p_risky,
        "p_cash": p_cash,
        "is_valid": 0 <= p_risky <= 1,
    }


def merton_optimal_allocation_multiasset(a, r, cov_matrix, gamma):
    """
    Analytical optimal allocation for n correlated risk assets + cash (unconstrained).
    Multi-asset Merton formula:

        p* = (1/gamma) * Sigma^{-1} * (mu - r * 1)

    where Sigma is the n×n covariance matrix of risky asset returns.

    Args:
        a: Array of expected returns, shape (n,)
        r: Risk-free rate (scalar)
        cov_matrix: Covariance matrix, shape (n, n)
        gamma: Absolute risk aversion coefficient

    Returns:
        dict with:
            p_risky: optimal risky allocations, shape (n,)
            p_cash: cash allocation (scalar)
            is_valid: True if all weights in [0,1] and sum ≤ 1
    """
    a = np.array(a)
    cov = np.array(cov_matrix)
    excess_return = a - r  # mu - r*1, shape (n,)

    try:
        cov_inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        return {"p_risky": None, "p_cash": None, "is_valid": False, "error": "Singular covariance matrix"}

    p_risky = (1.0 / gamma) * cov_inv @ excess_return  # shape (n,)
    p_cash = 1.0 - np.sum(p_risky)

    is_valid = bool(np.all(p_risky >= 0) and np.all(p_risky <= 1) and p_cash >= 0)

    return {
        "p_risky": p_risky,
        "p_cash": p_cash,
        "is_valid": is_valid,
    }


def normalize_portfolio(portfolio, method="sum_to_one"):
    """
    Normalize portfolio to ensure it sums to 1.
    """
    if method == "sum_to_one":
        total = np.sum(portfolio)
        if total > 0:
            return portfolio / total
        else:
            # Default: equal weight
            return np.ones_like(portfolio) / len(portfolio)
    return portfolio


def project_action_to_feasible_set(raw_action, current_portfolio, max_adjustment=0.1):
    """
    Project raw RL action onto the feasible rebalancing set.

    Three constraints enforced simultaneously:
      (A) Σ_k Δp_k = 0
            Budget-neutral: total bought = total sold (cash is just another asset).
      (B) Σ_{k: Δp_k > 0} Δp_k ≤ max_adjustment
            One-way turnover ≤ 10%.  This is the correct reading of "adjust at
            most 10% of your portfolio": you can move at most 10% of wealth from
            sellers to buyers.  With n=4 assets the old per-asset clip would have
            allowed 4×10% = 40% buys — 80% two-way turnover — badly wrong.
      (C) p_k + Δp_k ≥ 0  for all k
            No short-selling.

    Algorithm (O(n), converges in ≤ n+2 iterations):
      1. Center Δp so Σ=0  [satisfies A initially]
      2. Scale down if one-way turnover > max_adjustment  [satisfies B]
      3. Iteratively fix short positions:
           set Δp_k = −p_k for each over-sold k,
           reduce buys proportionally to restore Σ=0  [maintains A, may loosen B]
    """
    p = np.asarray(current_portfolio, dtype=np.float64)
    delta = np.asarray(raw_action, dtype=np.float64).copy()

    # (A) Project onto Σ=0 hyperplane
    delta -= delta.mean()

    # (B) Scale to one-way turnover constraint
    one_way = np.maximum(delta, 0.0).sum()
    if one_way > max_adjustment + 1e-9:
        delta *= max_adjustment / one_way

    # (C) Iteratively enforce no-shorting
    for _ in range(len(p) + 2):
        new_p = p + delta
        if new_p.min() >= -1e-9:
            break
        short_mask = new_p < 0
        # Bring over-sold positions to zero (reduce sell)
        delta[short_mask] = -p[short_mask]
        # Σ delta is now > 0; redistribute excess by scaling down buy positions
        excess = delta.sum()
        buy_mask = delta > 1e-12
        total_buy = delta[buy_mask].sum()
        if total_buy > 1e-12:
            delta[buy_mask] -= excess * delta[buy_mask] / total_buy
        else:
            delta[short_mask] = 0.0

    # Numerical cleanup: enforce exact Σ=0
    delta -= delta.sum() / len(delta)
    return delta


def simulate_one_step(wealth, portfolio, action, returns, r, max_adjustment=0.1):
    """
    Simulate one step of portfolio evolution.

    Args:
        wealth: Current wealth scalar
        portfolio: Current allocation [p_cash, p_asset1, ..., p_assetN]
        action: Raw action (adjustments)
        returns: Asset returns [r_asset1, ..., r_assetN]
        r: Risk-free rate
        max_adjustment: Max adjustment constraint

    Returns:
        new_wealth, new_portfolio
    """
    # Project action to feasible set
    feasible_action = project_action_to_feasible_set(action, portfolio, max_adjustment)

    # Apply action
    new_portfolio = portfolio + feasible_action
    new_portfolio = normalize_portfolio(new_portfolio)

    # Compute portfolio return
    portfolio_return = new_portfolio[0] * r + np.sum(new_portfolio[1:] * returns)

    # Update wealth
    new_wealth = wealth * (1 + portfolio_return)

    return new_wealth, new_portfolio


def reward_function(final_wealth, gamma):
    """
    Reward at terminal step: CARA utility of final wealth.
    """
    return cara_utility(final_wealth, gamma)
