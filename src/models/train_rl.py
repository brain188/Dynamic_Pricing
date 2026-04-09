from typing import List

import numpy as np
import torch.nn as nn
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.monitor import Monitor


class EpsilonGreedyBandit:
    def __init__(self, k=8, epsilon=1.0, epsilon_decay=0.9995, epsilon_min=0.05):
        self.k = k
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.q_values = np.zeros(k)  # Expected rewards
        self.counts = np.zeros(k)  # Number of times each arm was selected

    def select_arm(self) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(self.k)
        return np.argmax(self.q_values)

    def update(self, arm: int, reward: float):
        self.counts[arm] += 1
        # Incremental average update Rule: Q_new = Q_old + 1/n * (R - Q_old)
        self.q_values[arm] += (reward - self.q_values[arm]) / self.counts[arm]

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


class UCB1Bandit:
    def __init__(self, k=8, c=2.0):
        self.k = k
        self.c = c  # Exploration parameter

        self.q_values = np.zeros(k)
        self.counts = np.zeros(k)
        self.total_n = 0

    def select_arm(self) -> int:
        # Ensure all arms are selected at least once
        if self.total_n < self.k:
            return self.total_n

        # UCB Calculation: Q(a) + c * sqrt(ln(N) / n(a))
        ucb_values = self.q_values + self.c * np.sqrt(
            np.log(self.total_n) / self.counts
        )
        return np.argmax(ucb_values)

    def update(self, arm: int, reward: float):
        self.total_n += 1
        self.counts[arm] += 1
        self.q_values[arm] += (reward - self.q_values[arm]) / self.counts[arm]


class ThompsonSamplingBandit:
    def __init__(self, k=8):
        self.k = k
        # Beta distribution parameters (Successes, Failures)
        # We start with prior alpha=1, beta=1 (Uniform distribution)
        self.alphas = np.ones(k)
        self.betas = np.ones(k)

    def select_arm(self) -> int:
        # Sample from each arm's Beta distribution
        samples = np.random.beta(self.alphas, self.betas)
        return np.argmax(samples)

    def update(self, arm: int, success: bool):
        """
        Standard Thompson Sampling for Bernoulli rewards.
        In pricing, 'successful' means the ride was accepted.
        """
        if success:
            self.alphas[arm] += 1
        else:
            self.betas[arm] += 1


def run_bandit_experiment(env, bandit, multipliers: List[float], n_steps=10000) -> dict:
    """
    Runs a bandit experiment on the given environment.
    """
    rewards = []
    regrets = []
    selections = []
    cumulative_reward = 0
    cumulative_regret = 0

    obs, info = env.reset()

    for i in range(n_steps):
        # 1. Agent selects an arm
        arm = bandit.select_arm()
        multiplier = multipliers[arm]
        selections.append(arm)

        # 2. Oracle calculation for regret
        # We find the multiplier among our 8 that WOULD have
        # maximized the expected reward
        # Expected Reward = Price * P(Accept) * Forecasted Demand
        # Since Forecasted Demand and Reference Price are fixed for this step,
        # we just maximize Price * P(Accept)

        # We assume environmental variables (reference_price, beta)
        # are internal to env for this lookup
        # In this educational simulation, we reach into env state
        ref_price = 50.0  # From env.py
        env_beta = env.beta
        demand_signal = obs[5]

        expected_rewards = []
        for m in multipliers:
            p = m * ref_price
            p_acc = 1.0 / (1.0 + np.exp(env_beta * (p - ref_price)))
            expected_rewards.append(p * p_acc * demand_signal)

        optimal_reward = max(expected_rewards)

        # 3. Step in environment
        obs, reward, terminated, truncated, info = env.step([multiplier])

        # 4. Update bandit
        if isinstance(bandit, ThompsonSamplingBandit):
            bandit.update(arm, info["accepted"])
        else:
            bandit.update(arm, reward)

        regret = optimal_reward - reward
        cumulative_reward += reward
        cumulative_regret += regret

        rewards.append(cumulative_reward)
        regrets.append(cumulative_regret)

        if terminated or truncated:
            obs, info = env.reset()

    return {
        "cumulative_rewards": rewards,
        "cumulative_regret": regrets,
        "arm_counts": np.bincount(selections, minlength=len(multipliers)),
    }


class QLearningAgent:
    def __init__(
        self,
        n_actions=8,
        alpha=0.1,
        gamma=0.95,
        epsilon=1.0,
        epsilon_decay=0.999,
        epsilon_min=0.05,
    ):
        self.n_actions = n_actions
        self.n_states = 20  # 5 demand bins * 4 hour bins
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Q-Table
        self.q_table = np.zeros((self.n_states, n_actions))

    def discretize_state(self, obs: np.ndarray) -> int:
        """
        Maps continuous observation [ds_ratio, h_sin, h_cos, ...]
        to discrete index.
        Note: env observation vector is:
        [ds_ratio, h_sin, h_cos, d_sin, d_cos, forecast, loyalty, vehicle]
        We also have current_hour in the info or we can reconstruct it.
        Actually, let's look at the obs indices:
        0: demand_supply_ratio (~0.5 to 2.0)
        5: forecasted_demand (unused for simple binning)
        6: loyalty (for higher order, but let's stick to 20 states as per task)
        We'll use a virtual 'hour' from the sine/cosine if needed,
        but in our Env, we can just pass the hour or use the sin/cos to find it.
        Let's assume we pass the raw hour for simplicity if we can,
        or we use the ds_ratio and sin/cos.
        """
        ds_ratio = obs[0]
        # Hour recovery from sin/cos or just using the 0-24 logic
        # For simplicity, we'll assume the environment provides 'hour'
        # in info or we calculate it from sin/cos.
        # But wait, obs itself has sin/cos. Let's use h_sin/h_cos to get 0-23.
        h_sin, h_cos = obs[1], obs[2]
        hour = np.arctan2(h_sin, h_cos) * (24 / (2 * np.pi))
        if hour < 0:
            hour += 24

        # Binning ds_ratio into 5 (0.5 to 1.5+)
        ds_bins = np.linspace(0.6, 1.4, 4)  # 4 thresholds -> 5 bins
        ds_idx = np.searchsorted(ds_bins, ds_ratio)

        # Binning hour into 4 (0-6, 6-12, 12-18, 18-24)
        hour_bins = [6, 12, 18]
        hour_idx = np.searchsorted(hour_bins, hour)

        # Combined index: ds_idx (0-4) * 4 + hour_idx (0-3) = 0 to 19
        return ds_idx * 4 + hour_idx

    def select_action(self, state_idx: int) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.q_table[state_idx])

    def update(self, s: int, a: int, r: float, s_next: int):
        # Q(s,a) <- Q(s,a) + alpha * [r + gamma * max(Q(s',a')) - Q(s,a)]
        best_next_action = np.argmax(self.q_table[s_next])
        td_target = r + self.gamma * self.q_table[s_next][best_next_action]
        td_error = td_target - self.q_table[s][a]
        self.q_table[s][a] += self.alpha * td_error

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def train_q_learning(env, agent, n_episodes=5000) -> dict:
    """Trains a Q-Learning agent on the given environment."""
    performance = {"episode_rewards": [], "q_deltas": [], "epsilons": []}

    for ep in range(n_episodes):
        obs, info = env.reset()
        state_idx = agent.discretize_state(obs)
        total_reward = 0
        q_delta_sum = 0
        steps = 0

        done = False
        while not done:
            action_idx = agent.select_action(state_idx)
            # multipliers[action_idx] is handled inside the rollout loop or passed here?
            # Env expects multiplier in step([val])
            multiplier = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0][action_idx]

            obs_next, reward, terminated, truncated, info = env.step([multiplier])
            state_idx_next = agent.discretize_state(obs_next)

            # Record Q-value before update for convergence tracking
            old_q = agent.q_table[state_idx][action_idx]
            agent.update(state_idx, action_idx, reward, state_idx_next)
            q_delta_sum += abs(agent.q_table[state_idx][action_idx] - old_q)

            state_idx = state_idx_next
            total_reward += reward
            steps += 1
            done = terminated or truncated

        agent.decay_epsilon()
        performance["episode_rewards"].append(total_reward)
        performance["q_deltas"].append(q_delta_sum / max(1, steps))
        performance["epsilons"].append(agent.epsilon)

        if (ep + 1) % 500 == 0:
            print(
                f"Episode {ep+1}/{n_episodes} | "
                f"Avg Reward: {np.mean(performance['episode_rewards'][-100:]):.2f} | "
                f"Epsilon: {agent.epsilon:.3f}"
            )

    return performance


def train_ppo(env, total_timesteps=100000, log_dir="logs/ppo/") -> PPO:
    """Trains a PPO agent using Stable Baselines 3."""
    import os

    os.makedirs(log_dir, exist_ok=True)

    # Wrap environment for monitoring
    log_file = os.path.join(log_dir, "monitor.csv")
    monitored_env = Monitor(env, log_file)

    # Hyperparameters as per tasklist
    policy_kwargs = dict(
        net_arch=dict(pi=[128, 128], qf=[128, 128]),  # SB3 v2 syntax for net_arch
        activation_fn=nn.Tanh,
    )
    # Wait: SB3 net_arch for PPO is slightly different: [128, 128]
    # or dict(pi=[…], vf=[…])
    # Correction for PPO:
    policy_kwargs = dict(net_arch=[128, 128], activation_fn=nn.Tanh)

    model = PPO(
        "MlpPolicy",
        monitored_env,
        verbose=1,
        tensorboard_log=log_dir,
        clip_range=0.2,
        learning_rate=3e-4,
        n_steps=2048,
        ent_coef=0.01,
        policy_kwargs=policy_kwargs,
    )

    print(f"Starting PPO training for {total_timesteps} steps...")
    model.learn(total_timesteps=total_timesteps)
    return model


def train_a2c(env, total_timesteps=100000, log_dir="logs/a2c/") -> A2C:
    """Trains an A2C agent for comparison."""
    import os

    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "monitor.csv")
    monitored_env = Monitor(env, log_file)
    model = A2C("MlpPolicy", monitored_env, verbose=1)

    print(f"Starting A2C training for {total_timesteps} steps...")
    model.learn(total_timesteps=total_timesteps)
    return model


def evaluate_rl_policy(model, env, n_episodes=50) -> dict:
    """Evaluates an RL policy and returns KPI metrics."""
    all_episode_rewards = []
    all_fares = []
    all_acc_probs = []
    all_accepted = []

    for _ in range(n_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        while not done:
            # Handle different model types (SB3 vs custom)
            if hasattr(model, "predict"):
                action, _states = model.predict(obs, deterministic=True)
            else:
                # Assuming custom agent has select_action and handles discrete mapping
                # But for our eval, we'll just handle SB3 for now
                action = model.select_action(model.discretize_state(obs))
                # Map discrete index to multiplier
                action = [[0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0][action]]

            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward

            all_fares.append(info["fare"])
            all_acc_probs.append(info["acceptance_prob"])
            all_accepted.append(info["accepted"])

            done = terminated or truncated

        all_episode_rewards.append(episode_reward)

    return {
        "mean_reward": np.mean(all_episode_rewards),
        "mean_fare": np.mean(all_fares),
        "acceptance_rate": np.mean(all_accepted),
        "mean_acceptance_prob": np.mean(all_acc_probs),
    }
