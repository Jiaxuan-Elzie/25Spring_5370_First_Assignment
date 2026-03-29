import torch
import torch.optim as optim
import numpy as np

from .common import plot_training_history
from .method3_neural_actor_critic import NeuralGaussianPolicyB, NeuralValueCritic, evaluate_policy_detailed


def train_actor_critic_adjustable_lr(
    env,
    policy,
    critic,
    seed=42,
    n_episodes=2000,
    gamma=0.99,
    log_interval=100,
    initial_lr=1e-3,
    final_lr=1e-5,
    lr_decay_steps=1000,
):
    """
    Train an actor-critic agent with an adjustable learning rate schedule.

    Args:
        env: The environment to train in.
        policy (NeuralGaussianPolicyB): The actor model.
        critic (NeuralValueCritic): The critic model.
        seed (int): Random seed.
        n_episodes (int): Number of training episodes.
        gamma (float): Discount factor.
        log_interval (int): Interval for printing training progress.
        initial_lr (float): Starting learning rate for the optimizers.
        final_lr (float): Final learning rate to decay to.
        lr_decay_steps (int): Number of episodes over which to decay the learning rate.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    env.seed(seed)

    actor_optimizer = optim.Adam(policy.parameters(), lr=initial_lr)
    critic_optimizer = optim.Adam(critic.parameters(), lr=initial_lr)

    # Learning rate scheduler
    actor_lr_scheduler = optim.lr_scheduler.LambdaLR(
        actor_optimizer,
        lr_lambda=lambda episode: np.power(
            final_lr / initial_lr, min(episode, lr_decay_steps) / lr_decay_steps
        )
    )
    critic_lr_scheduler = optim.lr_scheduler.LambdaLR(
        critic_optimizer,
        lr_lambda=lambda episode: np.power(
            final_lr / initial_lr, min(episode, lr_decay_steps) / lr_decay_steps
        )
    )

    history = {
        "rewards": [],
        "actor_losses": [],
        "critic_losses": [],
        "learning_rates": [],
    }

    for i_episode in range(n_episodes):
        state = env.reset()
        log_probs = []
        values = []
        rewards = []
        masks = []
        entropy = 0

        # --- Collect trajectory ---
        for t in range(env.horizon):
            state_tensor = torch.from_numpy(state).float().unsqueeze(0)
            value = critic(state_tensor)
            action, log_prob, entropy_term = policy.act(state_tensor)

            next_state, reward, done, _ = env.step(action.cpu().numpy()[0])

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(torch.tensor([reward], dtype=torch.float))
            masks.append(torch.tensor([1 - done], dtype=torch.float))
            entropy += entropy_term
            state = next_state
            if done:
                break

        # --- Update ---
        next_state_tensor = torch.from_numpy(next_state).float().unsqueeze(0)
        next_value = critic(next_state_tensor)
        returns = compute_returns(next_value, rewards, masks, gamma)

        log_probs = torch.cat(log_probs)
        returns = torch.cat(returns).detach()
        values = torch.cat(values)

        advantage = returns - values

        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()

        actor_optimizer.zero_grad()
        critic_optimizer.zero_grad()

        actor_loss.backward()
        critic_loss.backward()

        actor_optimizer.step()
        critic_optimizer.step()
        
        # Step the schedulers
        actor_lr_scheduler.step()
        critic_lr_scheduler.step()

        # --- Logging ---
        history["rewards"].append(np.sum(r.item() for r in rewards))
        history["actor_losses"].append(actor_loss.item())
        history["critic_losses"].append(critic_loss.item())
        history["learning_rates"].append(actor_optimizer.param_groups[0]['lr'])

        if i_episode % log_interval == 0:
            print(
                f"Episode {i_episode}\t"
                f"Last reward: {history['rewards'][-1]:.2f}\t"
                f"Avg reward: {np.mean(history['rewards'][-log_interval:]):.2f}\t"
                f"LR: {history['learning_rates'][-1]:.6f}"
            )

    plot_training_history(history)
    return policy, critic, history

def compute_returns(next_value, rewards, masks, gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns
