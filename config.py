"""
Configuration for Asset Allocation RL Problem
"""

# MVP Config: Single Risk Asset + Cash, Time=1 (Sanity Check)
CONFIG_MVP_SANITY = {
    "n_assets": 1,                  # Risk assets (excluding cash)
    "T": 1,                         # Single period
    "r": 0.02,                      # Cash interest rate
    "a": [0.08],                    # Expected return of asset 1
    "s": [0.0016],                  # Variance of asset 1 (NOT std dev!)
    "gamma": 1.0,                   # Absolute risk aversion coefficient
    "initial_portfolio": [0.5, 0.5],  # [cash, asset1]
    "initial_wealth": 1.0,
    "max_portfolio_adjustment": 0.1,  # Max 10% adjustment per period
}

# MVP Config: Multi-period (T=5)
CONFIG_MVP_MULTIPERIOD = {
    "n_assets": 1,
    "T": 5,
    "r": 0.02,
    "a": [0.08],
    "s": [0.0016],
    "gamma": 1.0,
    "initial_portfolio": [0.5, 0.5],
    "initial_wealth": 1.0,
    "max_portfolio_adjustment": 0.1,
}

# Config: Two risk assets
CONFIG_TWO_ASSETS = {
    "n_assets": 2,
    "T": 5,
    "r": 0.02,
    "a": [0.08, 0.12],
    "s": [0.0016, 0.0081],  # Variance (std dev = 0.04, 0.09)
    "gamma": 1.0,
    "initial_portfolio": [0.2, 0.4, 0.4],  # [cash, asset1, asset2]
    "initial_wealth": 1.0,
    "max_portfolio_adjustment": 0.1,
}

# Config: Three risk assets (for n < 5 test)
CONFIG_THREE_ASSETS = {
    "n_assets": 3,
    "T": 7,
    "r": 0.02,
    "a": [0.06, 0.10, 0.15],
    "s": [0.0025, 0.01, 0.0225],  # Variance (std dev = 0.05, 0.10, 0.15)
    "gamma": 2.0,
    "initial_portfolio": [0.25, 0.25, 0.25, 0.25],
    "initial_wealth": 1.0,
    "max_portfolio_adjustment": 0.1,
}

# Config: Four risk assets (boundary test n < 5)
CONFIG_FOUR_ASSETS = {
    "n_assets": 4,
    "T": 9,
    "r": 0.03,
    "a": [0.05, 0.08, 0.12, 0.18],
    "s": [0.0016, 0.0036, 0.01, 0.0256],  # Variance
    "gamma": 1.5,
    "initial_portfolio": [0.2, 0.2, 0.2, 0.2, 0.2],
    "initial_wealth": 1.0,
    "max_portfolio_adjustment": 0.1,
}

# Training hyperparameters
TRAINING_CONFIG = {
    "n_episodes": 5000,
    "learning_rate": 3e-4,
    "gamma_discount": 0.99,  # Discount factor for RL (not risk aversion)
    "batch_size": 64,
    "hidden_dims": [128, 64],
    "seed": 42,
    "clip_ratio": 0.2,
}

# Test configurations for different scenarios
TEST_CONFIGS = {
    "sanity_check": CONFIG_MVP_SANITY,
    "multiperiod": CONFIG_MVP_MULTIPERIOD,
    "two_assets": CONFIG_TWO_ASSETS,
    "three_assets": CONFIG_THREE_ASSETS,
    "four_assets": CONFIG_FOUR_ASSETS,
}
