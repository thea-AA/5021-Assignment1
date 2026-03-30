# Assignment 1: RL-Based Discrete-Time Asset Allocation

# Group Member： YU XIAOYA, CAI XINYI

# Assignment report： [report.md]

## Project Structure

```
├── agent.py              # PPO agent implementation (custom)
├── env.py                # Asset allocation environment
├── config.py             # Configuration dictionaries
├── utils.py              # Utility functions (Merton solution, projection, etc.)
├── train.py              # Training loop with custom PPO
├── tests/
│   ├── test_comprehensive.py    # Comprehensive test suite (15 tests)
│   └── test_all_configs.py      # Train & evaluate on all configurations
├── outputs/              # Saved models and learning curves
├── report.md             # Detailed assignment report
└── assignment1.md        # Assignment description
```

## Running Commands

### Training

1. **Custom PPO implementation** :
   ```bash
   python train.py
   ```


### Testing

Two test suites are provided:

1. **Comprehensive test suite** (15 detailed tests):
   ```bash
   python tests/test_comprehensive.py
   ```

2. **All‑configurations test** (train and evaluate on 6 predefined + 5 random configurations):
   ```bash
   python tests/test_all_configs.py
   ```

The comprehensive test suite validates constraint satisfaction, time‑horizon robustness, asset‑number scalability, Merton comparison, parameter sensitivity, baseline comparisons, and extended market‑environment scenarios.
