"""
RL Agent for asset allocation using PPO
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal


class PolicyNetwork(nn.Module):
    """
    Actor network: maps state -> action distribution parameters
    """

    def __init__(self, state_dim, action_dim, hidden_dims=[128, 64]):
        super().__init__()

        layers = []
        prev_dim = state_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        self.net = nn.Sequential(*layers)

        # Output: mean of action distribution
        self.mu_head = nn.Linear(prev_dim, action_dim)

        # Log std (learnable parameter, initialized to -0.5 for stability)
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))

        # Initialize
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)

    def forward(self, state):
        """
        Args:
            state: (batch, state_dim)

        Returns:
            dist: torch.distributions.Normal
        """
        # Handle input
        if torch.isnan(state).any():
            print("WARNING: NaN in state input")

        features = self.net(state)
        if torch.isnan(features).any():
            print("WARNING: NaN in network features")
            features = torch.nan_to_num(features, nan=0.0)

        mu = self.mu_head(features)
        if torch.isnan(mu).any():
            print("WARNING: NaN in mu")
            mu = torch.nan_to_num(mu, nan=0.0)

        mu = torch.clamp(mu, -2, 2)

        log_std_clipped = torch.clamp(self.log_std, min=-1, max=1)
        std = torch.exp(log_std_clipped)
        std = torch.clamp(std, min=0.01, max=1.0)

        # Create distribution without validation
        dist = Normal(mu, std, validate_args=False)
        return dist

    def get_action_and_log_prob(self, state):
        """Sample action and compute log probability."""
        dist = self.forward(state)
        action = dist.rsample()
        action = torch.clamp(action, -1, 1)  # Ensure action is in valid range
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob


class ValueNetwork(nn.Module):
    """
    Critic network: maps state -> value
    """

    def __init__(self, state_dim, hidden_dims=[128, 64]):
        super().__init__()

        layers = []
        prev_dim = state_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

        # Initialize
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)

    def forward(self, state):
        """
        Args:
            state: (batch, state_dim)

        Returns:
            value: (batch, 1)
        """
        return self.net(state)


class PPOAgent:
    """
    Proximal Policy Optimization (PPO) agent with stability improvements.
    
    Enhancements:
    - Gradient clipping
    - Learning rate scheduling
    - Advantage normalization
    - Value loss clipping
    - Early stopping
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        lr=3e-4,
        gamma=0.99,  # RL discount factor
        hidden_dims=[128, 64],
        clip_ratio=0.2,
        device="cpu",
        max_grad_norm=0.5,
        use_lr_schedule=True,
        entropy_coef=0.01,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.max_grad_norm = max_grad_norm
        self.use_lr_schedule = use_lr_schedule
        self.entropy_coef = entropy_coef
        self.device = device

        self.policy = PolicyNetwork(state_dim, action_dim, hidden_dims).to(device)
        self.value_fn = ValueNetwork(state_dim, hidden_dims).to(device)

        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value_fn.parameters(), lr=lr)
        
        # Learning rate scheduler (optional)
        if use_lr_schedule:
            self.policy_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.policy_optimizer, mode='max', factor=0.5, patience=200, min_lr=1e-6
            )
            self.value_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.value_optimizer, mode='min', factor=0.5, patience=200, min_lr=1e-6
            )
        else:
            self.policy_scheduler = None
            self.value_scheduler = None
        
        # Training statistics
        self.training_stats = {
            "policy_loss": [],
            "value_loss": [],
            "mean_advantage": [],
            "grad_norm": [],
            "entropy": [],
        }

    def select_action(self, state):
        """
        Select action given state (for rollout).

        Args:
            state: (state_dim,) or (batch, state_dim)

        Returns:
            action: numpy array in [-1, 1]
        """
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(self.device)

        if state.dim() == 1:
            state = state.unsqueeze(0)

        with torch.no_grad():
            dist = self.policy.forward(state)
            action = dist.mean  # Use mean for deterministic action
            action = action.squeeze(0).cpu().numpy()

        return action

    def select_action_stochastic(self, state):
        """
        Select action stochastically (for training).
        """
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(self.device)

        if state.dim() == 1:
            state = state.unsqueeze(0)

        with torch.no_grad():
            action, log_prob = self.policy.get_action_and_log_prob(state)
            action = action.squeeze(0).cpu().numpy()

        return action

    def compute_advantage(self, rewards, values, dones):
        """
        Compute GAE (Generalized Advantage Estimation) for trajectory.

        Args:
            rewards: (T,)
            values: (T+1,)  includes bootstrap value
            dones: (T,)

        Returns:
            advantages: (T,)
            returns: (T,)
        """
        T = len(rewards)
        advantages = np.zeros(T)
        gae = 0

        for t in reversed(range(T)):
            # next_value is bootstrap value for terminal state
            next_value = values[t + 1] * (1 - dones[t])  # if done, next_value = 0
            delta = rewards[t] + self.gamma * next_value - values[t]
            gae = delta + self.gamma * 0.95 * (1 - dones[t]) * gae
            advantages[t] = gae

        returns = advantages + values[:T]
        return advantages, returns

    def update(self, states, actions, old_log_probs, advantages, returns, old_values=None, n_epochs=3):
        """
        Update policy and value network using PPO with stability improvements.

        Args:
            states: (batch_size, state_dim)
            actions: (batch_size, action_dim)
            old_log_probs: (batch_size,)
            advantages: (batch_size,)
            returns: (batch_size,)
            old_values: (batch_size,) old value predictions from rollout (optional, used for value clipping)
            n_epochs: Number of epochs per update
        """
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)

        # Use provided old_values or compute from returns - advantages
        if old_values is not None:
            old_values_tensor = torch.FloatTensor(old_values).to(self.device)
        else:
            # Fallback: compute from returns - advantages
            old_values_tensor = returns - advantages

        # Normalize advantages (critical for stability)
        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        epoch_losses = []
        
        for epoch in range(n_epochs):
            # Policy update
            dist = self.policy.forward(states)
            new_log_probs = dist.log_prob(actions).sum(dim=-1)
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # PPO surrogate loss
            surrogate1 = ratio * advantages
            surrogate2 = (
                torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
                * advantages
            )
            policy_loss = -torch.min(surrogate1, surrogate2).mean()

            # Entropy bonus for exploration
            entropy = dist.entropy().sum(dim=-1).mean()
            policy_loss = policy_loss - self.entropy_coef * entropy

            self.policy_optimizer.zero_grad()
            policy_loss.backward(retain_graph=True)
            
            # Gradient clipping (prevent explosion)
            policy_grad_norm = nn.utils.clip_grad_norm_(
                self.policy.parameters(), 
                max_norm=self.max_grad_norm
            )
            self.policy_optimizer.step()

            # Value network update
            values = self.value_fn.forward(states).squeeze(-1)
            
            # Clip value loss to prevent large updates (PPO-style)
            values_clipped = old_values_tensor + torch.clamp(values - old_values_tensor, -self.clip_ratio, self.clip_ratio)
            value_loss_unclipped = (values - returns) ** 2
            value_loss_clipped = (values_clipped - returns) ** 2
            value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()

            self.value_optimizer.zero_grad()
            value_loss.backward()
            
            value_grad_norm = nn.utils.clip_grad_norm_(
                self.value_fn.parameters(), 
                max_norm=self.max_grad_norm
            )
            self.value_optimizer.step()

            # Track losses
            epoch_losses.append({
                "policy": policy_loss.item(),
                "value": value_loss.item(),
                "policy_grad": policy_grad_norm.item(),
                "value_grad": value_grad_norm.item(),
                "entropy": entropy.item(),
            })

        # Store training statistics
        avg_policy_loss = np.mean([l["policy"] for l in epoch_losses])
        avg_value_loss = np.mean([l["value"] for l in epoch_losses])
        avg_entropy = np.mean([l["entropy"] for l in epoch_losses])

        self.training_stats["policy_loss"].append(avg_policy_loss)
        self.training_stats["value_loss"].append(avg_value_loss)
        self.training_stats["entropy"].append(avg_entropy)
        self.training_stats["grad_norm"].append(np.mean([l["policy_grad"] for l in epoch_losses]))
        self.training_stats["mean_advantage"].append(advantages.mean().item())

        # Update learning rate schedulers
        if self.policy_scheduler is not None:
            # Policy scheduler: mode='max', we want to maximize negative loss (minimize loss)
            self.policy_scheduler.step(-avg_policy_loss)
        if self.value_scheduler is not None:
            # Value scheduler: mode='min', we want to minimize value loss
            self.value_scheduler.step(avg_value_loss)

    def save_model(self, filepath):
        """Save policy and value networks."""
        torch.save(
            {
                "policy": self.policy.state_dict(),
                "value": self.value_fn.state_dict(),
            },
            filepath,
        )

    def load_model(self, filepath):
        """Load policy and value networks."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy"])
        self.value_fn.load_state_dict(checkpoint["value"])