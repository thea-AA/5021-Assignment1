"""
Configuration for Asset Allocation RL Problem
"""

# MVP Config: Single Risk Asset + Cash, Time=1 (Sanity Check)
CONFIG_MVP_SANITY = {
    "n_assets": 1,                  # Risk assets (excluding cash)
    "T": 1,                         # Single period
    "r": 0.02,                      # Cash interest rate
    "a": [0.08],                    # Expected return of asset 1
    "s": [0.04],                    # Variance (std dev) of asset 1
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
    "s": [0.04],
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
    "s": [0.04, 0.09],
    "gamma": 1.0,
    "initial_portfolio": [0.2, 0.4, 0.4],  # [cash, asset1, asset2]
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
}

# Test configurations for different scenarios
TEST_CONFIGS = {
    "sanity_check": CONFIG_MVP_SANITY,
    "multiperiod": CONFIG_MVP_MULTIPERIOD,
    "two_assets": CONFIG_TWO_ASSETS,
}
