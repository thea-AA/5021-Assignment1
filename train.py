"""
Training loop for asset allocation RL agent
"""
import numpy as np
import torch
from tqdm import tqdm

from config import CONFIG_MVP_SANITY, TRAINING_CONFIG
from env import AssetAllocationEnv
from agent import PPOAgent
from utils import merton_optimal_allocation


def rollout_episode(env, agent):
    """
    Collect one episode of experience.

    Returns:
        states, actions, rewards, log_probs, dones, values (including bootstrap)
    """
    states = []
    actions = []
    log_probs_list = []
    rewards = []
    dones = []
    values = []

    obs, _ = env.reset()
    done = False

    while not done:
        states.append(obs.copy())

        # Select action stochastically
        action = agent.select_action_stochastic(obs)
        actions.append(action)

        # Compute log prob (for PPO)
        state_tensor = torch.FloatTensor(obs).unsqueeze(0).to(agent.device)
        dist = agent.policy.forward(state_tensor)
        action_tensor = torch.FloatTensor(action).unsqueeze(0).to(agent.device)
        log_prob = dist.log_prob(action_tensor).sum(dim=-1).item()
        log_probs_list.append(log_prob)

        # Get value estimate
        with torch.no_grad():
            value = agent.value_fn.forward(state_tensor).item()
        values.append(value)

        # Step environment
        obs, reward, done, _, info = env.step(action)
        rewards.append(reward)
        dones.append(done)

    # Bootstrap value at terminal state
    if not done:
        state_tensor = torch.FloatTensor(obs).unsqueeze(0).to(agent.device)
        with torch.no_grad():
            bootstrap_value = agent.value_fn.forward(state_tensor).item()
    else:
        bootstrap_value = 0.0

    values.append(bootstrap_value)

    return {
        "states": np.array(states),
        "actions": np.array(actions),
        "log_probs": np.array(log_probs_list),
        "rewards": np.array(rewards),
        "dones": np.array(dones),
        "values": np.array(values),
    }


def train(env_config, train_config, n_episodes=None):
    """
    Train PPO agent.

    Args:
        env_config: Environment configuration dict
        train_config: Training configuration dict
        n_episodes: Number of episodes to train (override config)
    """
    if n_episodes is None:
        n_episodes = train_config["n_episodes"]

    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = AssetAllocationEnv(**env_config)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=train_config["learning_rate"],
        gamma=train_config["gamma_discount"],
        hidden_dims=train_config["hidden_dims"],
        clip_ratio=train_config.get("clip_ratio", 0.2),
        max_grad_norm=train_config.get("max_grad_norm", 0.5),
        use_lr_schedule=train_config.get("use_lr_schedule", True),
        entropy_coef=train_config.get("entropy_coef", 0.01),
        device=device,
    )

    np.random.seed(train_config["seed"])
    torch.manual_seed(train_config["seed"])

    # Training loop
    episode_rewards = []
    episode_final_wealths = []

    for episode in tqdm(range(n_episodes), desc="Training"):
        trajectory = rollout_episode(env, agent)

        states = trajectory["states"]
        actions = trajectory["actions"]
        log_probs = trajectory["log_probs"]
        rewards = trajectory["rewards"]
        dones = trajectory["dones"]
        values = trajectory["values"]

        # Compute advantages and returns
        advantages, returns = agent.compute_advantage(rewards, values, dones)

        # Update agent
        # values includes bootstrap value at the end, so old_values = values[:-1]
        old_values = values[:-1]
        agent.update(
            states=states,
            actions=actions,
            old_log_probs=log_probs,
            advantages=advantages,
            returns=returns,
            old_values=old_values,
            n_epochs=3,
        )

        # Track performance
        episode_reward = np.sum(rewards)
        final_wealth = env.wealth
        episode_rewards.append(episode_reward)
        episode_final_wealths.append(final_wealth)

        if (episode + 1) % 100 == 0:
            # Print training statistics
            if agent.training_stats["policy_loss"]:
                # Get the latest statistics (average over last update)
                latest_policy_loss = agent.training_stats["policy_loss"][-1]
                latest_value_loss = agent.training_stats["value_loss"][-1]
                latest_entropy = agent.training_stats["entropy"][-1] if agent.training_stats["entropy"] else 0.0
                latest_mean_advantage = agent.training_stats["mean_advantage"][-1]
                latest_grad_norm = agent.training_stats["grad_norm"][-1] if agent.training_stats["grad_norm"] else 0.0

                # Get current learning rate
                current_lr = agent.policy_optimizer.param_groups[0]['lr']

                # Calculate recent performance
                recent_rewards = episode_rewards[-100:] if len(episode_rewards) >= 100 else episode_rewards
                recent_wealths = episode_final_wealths[-100:] if len(episode_final_wealths) >= 100 else episode_final_wealths
                avg_reward = np.mean(recent_rewards) if recent_rewards else 0.0
                avg_wealth = np.mean(recent_wealths) if recent_wealths else 0.0

                print(
                    f"Episode {episode+1:4d}: "
                    f"Reward={avg_reward:.4f}, "
                    f"Wealth={avg_wealth:.4f}, "
                    f"Policy Loss={latest_policy_loss:.4f}, "
                    f"Value Loss={latest_value_loss:.4f}, "
                    f"Entropy={latest_entropy:.4f}, "
                    f"Advantage={latest_mean_advantage:.4f}, "
                    f"LR={current_lr:.6f}"
                )

    return agent, episode_rewards, episode_final_wealths


def evaluate_analytical(config):
    """
    Compare RL results with analytical solution (Merton).
    Only for n_assets=1, T=1, unconstrained case.
    """
    if config["n_assets"] != 1 or config["T"] != 1:
        print(
            "Analytical evaluation only available for n_assets=1, T=1"
        )
        return

    result = merton_optimal_allocation(
        a=config["a"][0],
        r=config["r"],
        s=config["s"][0],
        gamma=config["gamma"],
    )

    print("\n=== Analytical Solution (Merton) ===")
    print(f"Expected Return (a): {config['a'][0]:.4f}")
    print(f"Risk-free Rate (r): {config['r']:.4f}")
    print(f"Std Dev (s): {config['s'][0]:.4f}")
    print(f"Risk Aversion (γ): {config['gamma']:.4f}")
    print(f"\nOptimal Allocation:")
    print(f"  Cash (p_0): {result['p_cash']:.4f}")
    print(f"  Risk Asset (p_1): {result['p_risky']:.4f}")
    print(f"  Feasible (0 ≤ p_1 ≤ 1): {result['is_valid']}")

    return result


if __name__ == "__main__":
    # Example: Train on MVP sanity check config
    print("Starting training...")
    agent, rewards, wealths = train(
        env_config=CONFIG_MVP_SANITY,
        train_config=TRAINING_CONFIG,
        n_episodes=2000,
    )

    print("\n=== Training Results ===")
    print(f"Final avg reward (last 100): {np.mean(rewards[-100:]):.4f}")
    print(f"Final avg wealth (last 100): {np.mean(wealths[-100:]):.4f}")

    # Compare with analytical
    analytical_result = evaluate_analytical(CONFIG_MVP_SANITY)

    print("\n=== RL Learned Policy ===")
    env = AssetAllocationEnv(**CONFIG_MVP_SANITY)
    obs, _ = env.reset()
    action = agent.select_action(obs)
    print(f"Learned action (raw): {action}")
    print(f"Action scaled to adjustment: {action * CONFIG_MVP_SANITY['max_portfolio_adjustment']}")