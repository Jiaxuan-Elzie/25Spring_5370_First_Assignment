import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .common import (
    AssetAllocationEnv,
    apply_turnover_toward_target,
    build_static_env_config,
    build_static_env_from_config,
)


class NeuralGaussianPolicyB(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=32, init_std=0.20, lr=1e-3, seed=None):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.log_std = nn.Parameter(torch.ones(action_dim) * np.log(init_std))
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, state):
        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float32)
        if state.dim() == 1:
            state = state.unsqueeze(0)

        mu = self.net(state)
        log_std = torch.clamp(self.log_std, min=-4.0, max=1.0)
        std = torch.exp(log_std).unsqueeze(0).expand_as(mu)
        return mu, std

    def sample_action(self, state):
        if isinstance(state, np.ndarray):
            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        else:
            state_t = state.unsqueeze(0) if state.dim() == 1 else state

        mu, std = self.forward(state_t)
        dist = torch.distributions.Normal(mu, std)
        logits_t = dist.rsample()
        log_prob_t = dist.log_prob(logits_t.detach()).sum(dim=-1)
        entropy_t = dist.entropy().sum(dim=-1)
        target_w_t = F.softmax(logits_t, dim=-1)

        return {
            "logits": logits_t.squeeze(0),
            "log_prob": log_prob_t.squeeze(0),
            "entropy": entropy_t.squeeze(0),
            "mu": mu.squeeze(0),
            "std": std.squeeze(0),
            "target_w": target_w_t.squeeze(0),
        }

    def deterministic_action(self, state):
        with torch.no_grad():
            mu, _ = self.forward(state)
            target_w = F.softmax(mu, dim=-1)
        return {
            "mu": mu.squeeze(0).cpu().numpy(),
            "target_w": target_w.squeeze(0).cpu().numpy(),
        }

    def param_norm(self):
        total = 0.0
        with torch.no_grad():
            for parameter in self.parameters():
                total += torch.sum(parameter ** 2).item()
        return float(np.sqrt(total))


class NeuralValueCritic(nn.Module):
    def __init__(self, state_dim, hidden_dim=32, lr=1e-3, seed=None):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def forward(self, states):
        return self.net(states).squeeze(-1)

    def predict(self, state):
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            value = self.forward(state_t).item()
        return float(value)

    def update(self, state, target):
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        target_t = torch.tensor([target], dtype=torch.float32)
        pred = self.forward(state_t)
        loss = self.loss_fn(pred, target_t)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        td_error = target - pred.item()
        return float(loss.item()), float(td_error)

    def param_norm(self):
        total = 0.0
        with torch.no_grad():
            for parameter in self.parameters():
                total += torch.sum(parameter ** 2).item()
        return float(np.sqrt(total))


def evaluate_policy(env_config, policy, n_eval=200, max_turnover=0.10, seed=999, stochastic=False):
    utilities = []
    wealths = []

    for path_index in range(n_eval):
        env = build_static_env_from_config(env_config, seed + path_index)
        state = env.reset()
        done = False

        while not done:
            if stochastic:
                out = policy.sample_action(state)
                target_weights = out["target_w"].detach().cpu().numpy()
            else:
                out = policy.deterministic_action(state)
                target_weights = out["target_w"]

            action_weights = apply_turnover_toward_target(
                env.weights.copy(),
                target_weights,
                max_turnover=max_turnover,
            )
            state, reward, done, info = env.step(action_weights)

        utilities.append(reward)
        wealths.append(info["wealth"])

    return np.mean(utilities), np.std(utilities), np.mean(wealths), np.std(wealths)


def evaluate_policy_detailed(env_builder, policy, n_eval=200, max_turnover=0.10, seed=999, stochastic=False):
    utilities = []
    wealths = []
    all_exec_weights = []
    all_turnovers = []
    weights_regime0 = []
    weights_regime1 = []
    cond_correct = []

    for path_index in range(n_eval):
        env = env_builder(seed + path_index)
        state = env.reset()
        done = False

        while not done:
            if stochastic:
                out = policy.sample_action(state)
                target_weights = out["target_w"].detach().cpu().numpy()
            else:
                out = policy.deterministic_action(state)
                target_weights = out["target_w"]

            action_weights = apply_turnover_toward_target(
                env.weights.copy(),
                target_weights,
                max_turnover=max_turnover,
            )
            state, reward, done, info = env.step(action_weights)

            exec_w = info["executed_weights"]
            all_exec_weights.append(exec_w)
            all_turnovers.append(info["turnover"])
            regime = info.get("regime")
            if regime == 0:
                weights_regime0.append(exec_w)
                if len(exec_w) >= 3:
                    cond_correct.append(int(np.argmax(exec_w[1:]) == 0))
            elif regime == 1:
                weights_regime1.append(exec_w)
                if len(exec_w) >= 3:
                    cond_correct.append(int(np.argmax(exec_w[1:]) == 1))

        utilities.append(reward)
        wealths.append(info["wealth"])

    all_exec_weights = np.array(all_exec_weights)
    mean_weight = all_exec_weights.mean(axis=0)
    result = {
        "mean_utility": float(np.mean(utilities)),
        "std_utility": float(np.std(utilities)),
        "mean_wealth": float(np.mean(wealths)),
        "std_wealth": float(np.std(wealths)),
        "mean_weight": mean_weight,
        "avg_turnover": float(np.mean(all_turnovers)),
    }

    if len(mean_weight) >= 3:
        result["bias_to_risky1"] = float(mean_weight[1] - np.mean(mean_weight[2:])) if len(mean_weight) > 2 else float(mean_weight[1])

    if weights_regime0 and weights_regime1:
        expected_w_regime0 = np.mean(np.array(weights_regime0), axis=0)
        expected_w_regime1 = np.mean(np.array(weights_regime1), axis=0)
        result["Ew_regime0"] = expected_w_regime0
        result["Ew_regime1"] = expected_w_regime1

        if len(expected_w_regime0) >= 3:
            result["state_response_asset1"] = float(expected_w_regime0[1] - expected_w_regime1[1])
            result["state_response_asset2"] = float(expected_w_regime1[2] - expected_w_regime0[2]) if len(expected_w_regime0) > 2 else np.nan

        result["conditional_accuracy"] = float(np.mean(cond_correct)) if cond_correct else np.nan

    return result


def train_actor_critic(
    env,
    episodes=5000,
    policy_lr=1e-3,
    critic_lr=1e-3,
    policy_hidden_dim=32,
    critic_hidden_dim=32,
    gamma=1.0,
    max_turnover=0.10,
    eval_every=100,
    eval_paths=200,
    seed=None,
    entropy_coef=1e-3,
    advantage_clip=5.0,
    eval_env_builder=None,
):
    state_dim = len(env.reset())
    action_dim = env.n_assets_total

    policy = NeuralGaussianPolicyB(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=policy_hidden_dim,
        init_std=0.20,
        lr=policy_lr,
        seed=seed,
    )
    critic = NeuralValueCritic(
        state_dim=state_dim,
        hidden_dim=critic_hidden_dim,
        lr=critic_lr,
        seed=seed,
    )

    static_env_config = build_static_env_config(env) if isinstance(env, AssetAllocationEnv) else None
    history = {
        "value_type": "neural",
        "train_utility": [],
        "train_terminal_wealth": [],
        "policy_norm": [],
        "value_norm": [],
        "value_loss": [],
        "eval_episode": [],
        "eval_mean_utility": [],
        "eval_std_utility": [],
        "eval_mean_wealth": [],
        "eval_std_wealth": [],
    }

    for episode in range(episodes):
        state = env.reset()
        done = False
        episode_reward = 0.0
        terminal_wealth = np.nan
        step_critic_losses = []

        while not done:
            out = policy.sample_action(state)
            log_prob = out["log_prob"]
            entropy = out["entropy"]
            target_weights = out["target_w"].detach().cpu().numpy()

            new_weights = apply_turnover_toward_target(
                env.weights.copy(),
                target_weights,
                max_turnover=max_turnover,
            )
            next_state, reward, done, info = env.step(new_weights)

            if done:
                td_target = reward
            else:
                td_target = reward + gamma * critic.predict(next_state)

            critic_loss, td_error = critic.update(state, td_target)
            advantage = np.clip(td_error, -advantage_clip, advantage_clip)
            advantage_t = torch.tensor(advantage, dtype=torch.float32)

            actor_loss = -(log_prob * advantage_t) - entropy_coef * entropy
            policy.optimizer.zero_grad()
            actor_loss.backward()
            policy.optimizer.step()

            step_critic_losses.append(critic_loss)
            episode_reward += reward
            terminal_wealth = info["wealth"]
            state = next_state

        history["train_utility"].append(episode_reward)
        history["train_terminal_wealth"].append(terminal_wealth)
        history["policy_norm"].append(policy.param_norm())
        history["value_norm"].append(critic.param_norm())
        history["value_loss"].append(np.mean(step_critic_losses) if step_critic_losses else 0.0)

        if (episode + 1) % eval_every == 0:
            if eval_env_builder is not None:
                metrics = evaluate_policy_detailed(
                    env_builder=eval_env_builder,
                    policy=policy,
                    n_eval=eval_paths,
                    max_turnover=max_turnover,
                    seed=999,
                    stochastic=False,
                )
                mean_u = metrics["mean_utility"]
                std_u = metrics["std_utility"]
                mean_w = metrics["mean_wealth"]
                std_w = metrics["std_wealth"]
            elif static_env_config is not None:
                mean_u, std_u, mean_w, std_w = evaluate_policy(
                    static_env_config,
                    policy,
                    n_eval=eval_paths,
                    max_turnover=max_turnover,
                    seed=999,
                    stochastic=False,
                )
            else:
                raise ValueError("eval_env_builder is required for non-static environments.")

            history["eval_episode"].append(episode + 1)
            history["eval_mean_utility"].append(mean_u)
            history["eval_std_utility"].append(std_u)
            history["eval_mean_wealth"].append(mean_w)
            history["eval_std_wealth"].append(std_w)

            print(
                f"[scheme B] Episode {episode + 1:4d} | "
                f"Train utility (last100 mean) = {np.mean(history['train_utility'][-100:]):.6f} | "
                f"Critic loss = {np.mean(history['value_loss'][-100:]):.6f} | "
                f"Eval mean utility = {mean_u:.6f} | "
                f"Eval mean wealth = {mean_w:.4f}"
            )

    return policy, critic, history
