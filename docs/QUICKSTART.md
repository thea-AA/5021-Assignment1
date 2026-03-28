# Quick Start Guide - Assignment 1: RL Asset Allocation

## 5分钟快速开始

### 1️⃣ 安装
```bash
pip install gymnasium torch stable-baselines3 numpy matplotlib tqdm
```

### 2️⃣ 验证安装
```bash
python test_mvp.py
```
✅ 应看到：测试通过，环境动态正确，约束满足

### 3️⃣ 查看演示
```bash
python demo.py
```
✅ 应看到：环境演示、分析解、风险敏感性分析、可扩展性测试

### 4️⃣ 训练模型
```bash
python train_sb3.py
```
⏱️ 耗时：~2-3分钟

✅ 输出：
- 训练进度 (episodes)
- 学习曲线 (learning_curve.png)
- 训练后模型 (asset_allocation_ppo.zip)
- 性能评估结果

---

## 核心文件速查

| 文件 | 用途 | 关键类/函数 |
|------|------|-----------|
| `config.py` | 参数配置 | `CONFIG_MVP_SANITY`, `CONFIG_TWO_ASSETS` |
| `env.py` | 环境 | `AssetAllocationEnv` |
| `agent.py` | 神经网络 | `PolicyNetwork`, `ValueNetwork` |
| `train_sb3.py` | 训练（推荐） | `train_sb3()`, `evaluate_policy()` |
| `utils.py` | 工具函数 | `cara_utility()`, `merton_optimal_allocation()` |
| `test_mvp.py` | 测试 | 5个单元测试 |

---

## 常见用法

### 用法1：修改参数训练
```python
from config import TRAINING_CONFIG
from train_sb3 import train_sb3

config = {
    "n_assets": 2,  # 2个风险资产
    "T": 5,         # 5期
    "r": 0.02,      # 2%无风险率
    "a": [0.08, 0.12],
    "s": [0.04, 0.08],
    "gamma": 1.5,   # 风险厌恶系数
    "initial_portfolio": [0.2, 0.4, 0.4],
    "initial_wealth": 1.0,
    "max_portfolio_adjustment": 0.1,
}

model, rewards = train_sb3(config, n_steps=50000)
```

### 用法2：加载预训练模型进行推理
```python
from stable_baselines3 import PPO
from env import AssetAllocationEnv
from config import CONFIG_MVP_SANITY

model = PPO.load("asset_allocation_ppo")
env = AssetAllocationEnv(**CONFIG_MVP_SANITY)

obs, _ = env.reset()
action, _ = model.predict(obs, deterministic=True)
print(f"Action: {action}")
```

### 用法3：对标分析解
```python
from utils import merton_optimal_allocation

result = merton_optimal_allocation(
    a=0.08,
    r=0.02,
    s=0.04,
    gamma=1.0
)
print(f"Optimal risky allocation: {result['p_risky']:.4f}")
```

### 用法4：自定义环境测试
```python
from env import AssetAllocationEnv
import numpy as np

env = AssetAllocationEnv(
    n_assets=1, T=1, r=0.02,
    a=[0.08], s=[0.04], gamma=1.0,
    initial_portfolio=[0.5, 0.5]
)

obs, _ = env.reset()
for _ in range(5):
    action = np.random.uniform(-1, 1, size=2)
    obs, reward, done, _, info = env.step(action)
    print(f"Wealth: {info['wealth']:.4f}, Reward: {reward:.6f}")
```

---

## 配置预设

### 简单（推荐新手）
```python
from config import CONFIG_MVP_SANITY
# n=1资产, T=1期, 快速训练
```

### 中等难度
```python
from config import CONFIG_MVP_MULTIPERIOD
# n=1资产, T=5期, 学习时间策略
```

### 高级
```python
from config import CONFIG_TWO_ASSETS
# n=2资产, T=5期, 多资产决策
```

---

## 常见问题

**Q1: 训练太慢？**
- 减少 `n_steps` 参数 (e.g., 10000 instead of 50000)
- 使用GPU: `device='cuda'` in PPO初始化

**Q2: 结果不稳定？**
- 增加训练步数
- 固定随机种子: `model = PPO(..., seed=42)`
- 尝试不同学习率

**Q3: 如何用多个GPU？**
- 使用 `n_envs > 1`: `make_vec_env(lambda: env, n_envs=4)`

**Q4: 如何导出训练结果？**
```python
import numpy as np

# 保存学习曲线
np.save("rewards.npy", rewards)

# 保存模型
model.save("my_model")

# 加载模型
model = PPO.load("my_model")
```

---

## 验证清单

运行以下命令验证全部组件：

```bash
# ✅ 测试1: 环境和约束
python -c "from test_mvp import test_environment, test_action_projection; test_environment(); test_action_projection()"

# ✅ 测试2: 分析解
python -c "from test_mvp import test_analytical_solution; test_analytical_solution()"

# ✅ 测试3: PPO代理
python -c "from test_mvp import test_ppo_agent; test_ppo_agent()"

# ✅ 测试4: 快速训练
python -c "from test_mvp import test_quick_train; test_quick_train()"

# ✅ 完整演示
python demo.py

# ✅ 完整训练
python train_sb3.py
```

---

## 输出文件

训练后生成：
- `learning_curve.png` - 学习曲线
- `asset_allocation_ppo.zip` - 训练好的模型
- 控制台输出 - 性能指标

---

## 下一步

1. **理解代码** → 阅读 `IMPLEMENTATION_SUMMARY.md`
2. **修改参数** → 尝试不同的 `n`, `T`, `γ` 值
3. **扩展功能** → 添加更多资产、制约、或不同RL算法
4. **可视化** → 绘制策略热力图、财富分布等

---

## 支持的环境

- 🐍 Python 3.8+
- 🖥️ macOS / Linux / Windows
- 💾 最少需求：4GB RAM, 1GB 硬盘空间
- ⚡ 推荐：GPU加速（可选）

---

**开始探索吧！** 🚀
