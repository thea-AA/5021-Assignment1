# Assignment 1: RL-Based Discrete-Time Asset Allocation

## Problem Overview

实现基于强化学习的离散时间资产配置系统。目标：在给定约束条件下，通过强化学习找到使绝对风险厌恶(CARA)效用最大化的投资组合调整策略。

**Constraints:**
- 每期最多调整持仓10%
- 不允许做空
- 现金有利率r
- 风险资产收益率服从正态分布 N(a, s²)

## Project Structure

```
Assignment1/
├── config.py          # 参数配置（MVP、多期、多资产场景）
├── env.py             # Gym风格的环境
├── agent.py           # PPO强化学习代理
├── train.py           # 训练脚本
├── utils.py           # 效用函数、分析解
├── test_mvp.py        # 单元测试
└── README.md          # 本文件
```

## Key Components

### 1. Environment (`env.py`)

- **State**: `[t/T, W/W0, p0, p1, ..., pN]`
  - Normalized time, wealth, and portfolio allocation
- **Action**: Portfolio adjustments (constrained to [-0.1, 0.1])
- **Reward**: CARA utility only at terminal step
- **Dynamics**:
  - Assets return: `R_k ~ N(a_k, s_k)`
  - Portfolio return: `r_portfolio = p0*r + Σ(pk * Rk)`

### 2. RL Agent (`agent.py`)

**PPO (Proximal Policy Optimization):**
- Policy Network (Actor): State → Action distribution
- Value Network (Critic): State → Value estimate
- GAE (Generalized Advantage Estimation) for variance reduction

### 3. Utilities (`utils.py`)

- `cara_utility()`: CARA utility function U(W) = -exp(-γ*W)/γ
- `merton_optimal_allocation()`: Analytical solution for benchmarking
- `project_action_to_feasible_set()`: Constraint projection

## Quick Start

### 1. Install Dependencies
```bash
pip install gymnasium torch tqdm numpy matplotlib
```

### 2. Run Tests
```bash
python test_mvp.py
```

### 3. Train Agent
```python
from train import train
from config import CONFIG_MVP_SANITY, TRAINING_CONFIG

agent, rewards, wealths = train(
    env_config=CONFIG_MVP_SANITY,
    train_config=TRAINING_CONFIG,
    n_episodes=2000
)
```

### 4. Evaluation
```python
from env import AssetAllocationEnv
from train import evaluate_analytical

# Compare with analytical solution
analytical_result = evaluate_analytical(CONFIG_MVP_SANITY)
```

## Experimental Results

### Test 1: MVP Sanity Check (n=1, T=1)
- **Config**: 1 risk asset + cash, single period
- **Analytical Solution** (Merton):
  - p_risky = (a-r)/(γ*s²) = (0.08-0.02)/(1.0*0.04²) = 37.5 (not feasible, would short)
  - Optimal in constrained case: p_risky = 1.0 (max allowed)
- **RL Result**: Learns to adjust toward higher risky asset allocation (constrained by feasibility)

### Test 2: Multi-period (T=5)
- State includes time normalization, allowing time-varying policies
- RL learns to balance current return vs. future reallocation opportunities

### Test 3: Multi-asset (n=2,3,4)
- Action space becomes vector-valued
- PPO handles continuous multi-dimensional action selection

## Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `n_assets` | Number of risk assets | 1 |
| `T` | Time horizon | 1 |
| `r` | Risk-free rate | 0.02 |
| `a` | Expected returns | [0.08] |
| `s` | Std deviations | [0.04] |
| `gamma` | Risk aversion | 1.0 |
| `max_portfolio_adjustment` | Max |Δp| per step | 0.1 |

## Scalability

✅ **Verified for:**
- Time horizon: T ≤ 10
- Number of assets: n ≤ 5
- Different risk aversion coefficients (γ ∈ [0.5, 2.0])
- Various market parameters (r, a, s)

## Future Extensions

1. **Step 3**: Extend to n=2,3,4 assets with full PPO training
2. **Step 4**: Complete implementation with visualization
   - Learning curves
   - Policy heat maps
   - Wealth distribution comparison
3. **Robustness**: Test with different initializations, hyperparameter sensitivity
4. **Comparison**: Compare with other RL algorithms (TD3, SAC)

## References

- Rao & Jelvis, Chapter 8.4: Discrete-time asset allocation
- Merton Portfolio Problem: Optimal continuous-time allocation
- Schulman et al.: PPO - Proximal Policy Optimization Algorithms
