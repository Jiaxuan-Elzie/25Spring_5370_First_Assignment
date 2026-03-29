import matplotlib.pyplot as plt
import numpy as np

from asset_allocation.common import AssetAllocationEnv, plot_eval_wealth, plot_training_history
from asset_allocation.experiment_envs import SingleAdvantageEnv, TwoStateRotationEnv
from asset_allocation.method3_neural_actor_critic import evaluate_policy_detailed, train_actor_critic


def plot_turnover_scan_results(results):
    max_turnovers = [row["max_turnover"] for row in results]
    avg_turnovers = [row["avg_turnover"] for row in results]
    mean_utilities = [row["mean_utility"] for row in results]
    conditional_accuracies = [row["conditional_accuracy"] for row in results]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    axes[0].plot(max_turnovers, avg_turnovers, marker="o")
    axes[0].set_title("Max turnover vs realized avg turnover")
    axes[0].set_xlabel("max_turnover")
    axes[0].set_ylabel("avg turnover")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(max_turnovers, mean_utilities, marker="o")
    axes[1].set_title("Max turnover vs mean utility")
    axes[1].set_xlabel("max_turnover")
    axes[1].set_ylabel("mean utility")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(max_turnovers, conditional_accuracies, marker="o")
    axes[2].set_title("Max turnover vs conditional accuracy")
    axes[2].set_xlabel("max_turnover")
    axes[2].set_ylabel("conditional accuracy")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def build_static_market_env(seed):
    return AssetAllocationEnv(
        means=[0.08, 0.12, 0.05],
        variances=[0.04, 0.09, 0.02],
        cash_rate=0.03,
        horizon=6,
        init_weights=[0.25, 0.25, 0.30, 0.20],
        init_wealth=1.0,
        risk_aversion=1.5,
        seed=seed,
    )


def build_single_advantage_env(seed):
    return SingleAdvantageEnv(
        n_risky=3,
        mu_good=0.12,
        mu_bad=0.03,
        sigma_good=0.15,
        sigma_bad=0.15,
        cash_rate=0.02,
        horizon=6,
        init_weights=[0.25, 0.25, 0.25, 0.25],
        init_wealth=1.0,
        risk_aversion=1.5,
        seed=seed,
    )


def build_two_state_rotation_env(seed):
    return TwoStateRotationEnv(
        n_risky=3,
        mu_hi=0.15,
        mu_lo=-0.02,
        sigma=0.12,
        cash_rate=0.02,
        horizon=6,
        stay_prob=0.9,
        init_weights=[0.25, 0.25, 0.25, 0.25],
        init_wealth=1.0,
        risk_aversion=1.5,
        seed=seed,
    )


def run_static_market_experiment(episodes=3000, seed=123):
    env = build_static_market_env(seed)
    policy, critic, history = train_actor_critic(
        env,
        episodes=episodes,
        policy_lr=3e-4,
        critic_lr=1e-4,
        policy_hidden_dim=32,
        critic_hidden_dim=32,
        gamma=1.0,
        max_turnover=0.10,
        eval_every=100,
        eval_paths=200,
        seed=seed,
        entropy_coef=5e-4,
        advantage_clip=5.0,
    )
    return policy, critic, history


def run_single_advantage_experiment(episodes=3000, max_turnover=0.10, seed=123):
    env = build_single_advantage_env(seed)
    policy, critic, history = train_actor_critic(
        env,
        episodes=episodes,
        policy_lr=3e-4,
        critic_lr=1e-4,
        policy_hidden_dim=32,
        critic_hidden_dim=32,
        gamma=1.0,
        max_turnover=max_turnover,
        eval_every=100,
        eval_paths=200,
        seed=seed,
        entropy_coef=5e-4,
        advantage_clip=5.0,
        eval_env_builder=build_single_advantage_env,
    )

    metrics = evaluate_policy_detailed(
        env_builder=build_single_advantage_env,
        policy=policy,
        n_eval=300,
        max_turnover=max_turnover,
        seed=999,
        stochastic=False,
    )

    print("\n[Experiment 2.1] Single Advantage Market")
    print("Mean utility:", metrics["mean_utility"])
    print("Std utility:", metrics["std_utility"])
    print("Mean wealth:", metrics["mean_wealth"])
    print("Std wealth:", metrics["std_wealth"])
    print("Mean weight:", metrics["mean_weight"])
    print("Avg turnover:", metrics["avg_turnover"])
    print("Bias to risky asset 1:", metrics.get("bias_to_risky1"))

    return policy, critic, history, metrics


def run_two_state_rotation_experiment(episodes=4000, max_turnover=0.10, seed=123):
    env = build_two_state_rotation_env(seed)
    policy, critic, history = train_actor_critic(
        env,
        episodes=episodes,
        policy_lr=3e-4,
        critic_lr=1e-4,
        policy_hidden_dim=32,
        critic_hidden_dim=32,
        gamma=1.0,
        max_turnover=max_turnover,
        eval_every=100,
        eval_paths=200,
        seed=seed,
        entropy_coef=5e-4,
        advantage_clip=5.0,
        eval_env_builder=build_two_state_rotation_env,
    )

    metrics = evaluate_policy_detailed(
        env_builder=build_two_state_rotation_env,
        policy=policy,
        n_eval=300,
        max_turnover=max_turnover,
        seed=999,
        stochastic=False,
    )

    print("\n[Experiment 2.2] Two-State Rotation Market")
    print("Mean utility:", metrics["mean_utility"])
    print("Std utility:", metrics["std_utility"])
    print("Mean wealth:", metrics["mean_wealth"])
    print("Std wealth:", metrics["std_wealth"])
    print("Mean weight:", metrics["mean_weight"])
    print("Avg turnover:", metrics["avg_turnover"])
    print("Conditional accuracy:", metrics.get("conditional_accuracy"))
    print("State response asset1:", metrics.get("state_response_asset1"))
    print("State response asset2:", metrics.get("state_response_asset2"))
    print("E[w | regime 0]:", metrics.get("Ew_regime0"))
    print("E[w | regime 1]:", metrics.get("Ew_regime1"))

    return policy, critic, history, metrics


def run_turnover_scan_experiment(turnovers=(0.20, 0.10, 0.05, 0.02), episodes=3000, seed=123):
    results = []

    for max_turnover in turnovers:
        print("\n" + "=" * 80)
        print(f"Running turnover-friction scan: max_turnover = {max_turnover}")

        env = build_two_state_rotation_env(seed)
        policy, critic, history = train_actor_critic(
            env,
            episodes=episodes,
            policy_lr=3e-4,
            critic_lr=1e-4,
            policy_hidden_dim=32,
            critic_hidden_dim=32,
            gamma=1.0,
            max_turnover=max_turnover,
            eval_every=100,
            eval_paths=200,
            seed=seed,
            entropy_coef=5e-4,
            advantage_clip=5.0,
            eval_env_builder=build_two_state_rotation_env,
        )

        metrics = evaluate_policy_detailed(
            env_builder=build_two_state_rotation_env,
            policy=policy,
            n_eval=300,
            max_turnover=max_turnover,
            seed=999,
            stochastic=False,
        )

        row = {
            "max_turnover": max_turnover,
            "mean_utility": metrics["mean_utility"],
            "mean_wealth": metrics["mean_wealth"],
            "avg_turnover": metrics["avg_turnover"],
            "conditional_accuracy": metrics.get("conditional_accuracy", np.nan),
            "state_response_asset1": metrics.get("state_response_asset1", np.nan),
            "state_response_asset2": metrics.get("state_response_asset2", np.nan),
        }
        results.append(row)
        print("Result:", row)

    return results


if __name__ == "__main__":
    mode = "single_advantage"

    if mode == "static":
        policy, critic, history = run_static_market_experiment(episodes=3000, seed=123)
        print("=" * 80)
        print("Finished training on static market.")
        plot_training_history(history, ma_window=100, value_label="Critic")
        plot_eval_wealth(history, value_label="Critic")

    elif mode == "single_advantage":
        policy, critic, history, metrics = run_single_advantage_experiment(
            episodes=3000,
            max_turnover=0.10,
            seed=123,
        )
        print("=" * 80)
        print("Finished Experiment 2.1: Single Advantage Market")
        plot_training_history(history, ma_window=100, value_label="Critic")
        plot_eval_wealth(history, value_label="Critic")

    elif mode == "two_state":
        policy, critic, history, metrics = run_two_state_rotation_experiment(
            episodes=4000,
            max_turnover=0.10,
            seed=123,
        )
        print("=" * 80)
        print("Finished Experiment 2.2: Two-State Rotation Market")
        plot_training_history(history, ma_window=100, value_label="Critic")
        plot_eval_wealth(history, value_label="Critic")

    elif mode == "turnover_scan":
        results = run_turnover_scan_experiment(
            turnovers=(0.20, 0.10, 0.05, 0.02),
            episodes=3000,
            seed=123,
        )
        print("\nFinal scan results:")
        for row in results:
            print(row)
        plot_turnover_scan_results(results)

    else:
        raise ValueError(f"Unknown mode: {mode}")
