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
        # Clip to avoid overflow
        exponent = np.clip(-gamma * wealth, -100, 100)
        return -np.exp(exponent) / gamma
    except:
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
        # Avoid division by zero
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
        "is_valid": 0 <= p_risky <= 1,  # Check if unconstrained solution is feasible
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


def project_action_to_feasible_set(action, current_portfolio, max_adjustment=0.1):
    """
    Project action (portfolio adjustment) to feasible set.
    Constraints:
    1. |Δp_k| ≤ max_adjustment
    2. sum(Δp) = 0 (total portfolio stays at 1)
    3. p_new = p_current + Δp ≥ 0 (no short selling)

    Args:
        action: Raw action (unconstrained adjustment)
        current_portfolio: Current portfolio allocation
        max_adjustment: Maximum adjustment per period

    Returns:
        Feasible action that satisfies all constraints
    """
    # Step 1: Clip to [-max_adjustment, max_adjustment]
    clipped_action = np.clip(action, -max_adjustment, max_adjustment)

    # Step 2: Ensure new portfolio is non-negative
    new_portfolio = current_portfolio + clipped_action
    negative_mask = new_portfolio < 0
    clipped_action[negative_mask] = -current_portfolio[negative_mask]

    # Step 3: Adjust to ensure sum(action) = 0
    # Move excess to cash (index 0)
    action_sum = np.sum(clipped_action)
    clipped_action[0] -= action_sum  # Adjust cash position

    # Ensure cash doesn't go negative
    if clipped_action[0] + current_portfolio[0] < 0:
        clipped_action[0] = -current_portfolio[0]

    return clipped_action


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
