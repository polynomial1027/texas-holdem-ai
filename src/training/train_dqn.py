import os
from statistics import mean

from src.holdem.game import create_default_game
from src.holdem.encoder import encode_observation
from src.agents.dqn_agent import DQNAgent
from src.agents.random_agent import RandomAgent


def train(
    num_episodes=1000,
    save_interval=200,
):
    """
    最小 DQN 训练循环。

    Player 1: DQNAgent
    Player 2: RandomAgent

    注意：
    当前 reward 仍然是非常简化的版本。
    这个脚本的目标是先跑通训练流程。
    """

    os.makedirs("models", exist_ok=True)

    game = create_default_game(num_players=2)

    dqn_agent = DQNAgent()
    random_agent = RandomAgent()

    episode_rewards = []
    episode_losses = []
    win_count = 0

    for episode in range(1, num_episodes + 1):
        observation = game.reset()
        done = False

        total_reward_for_dqn = 0
        losses_this_episode = []

        while not done:
            current_player = observation["current_player"]

            if current_player == "Player 1":
                state = encode_observation(observation)
                legal_actions = observation["legal_actions"]

                action, action_index = dqn_agent.choose_action(
                    state,
                    legal_actions,
                )

                next_observation, reward, done, info = game.step(action)

                next_state = encode_observation(next_observation)
                next_legal_actions = next_observation["legal_actions"]

                dqn_agent.store_transition(
                    state=state,
                    action_index=action_index,
                    reward=reward,
                    next_state=next_state,
                    done=done,
                    next_legal_actions=next_legal_actions,
                )

                loss = dqn_agent.learn()

                if loss is not None:
                    losses_this_episode.append(loss)

                total_reward_for_dqn += reward
                observation = next_observation

            else:
                action = random_agent.choose_action(observation)
                observation, reward, done, info = game.step(action)

        dqn_agent.decay_epsilon()

        if game.winners and game.winners[0].name == "Player 1":
            win_count += 1

        episode_rewards.append(total_reward_for_dqn)

        if losses_this_episode:
            episode_losses.append(mean(losses_this_episode))

        if episode % 50 == 0:
            recent_rewards = episode_rewards[-50:]
            recent_losses = episode_losses[-50:] if episode_losses else [0.0]
            recent_win_rate = win_count / episode

            print(
                f"Episode {episode:5d} | "
                f"Avg Reward: {mean(recent_rewards):8.3f} | "
                f"Avg Loss: {mean(recent_losses):8.5f} | "
                f"Epsilon: {dqn_agent.epsilon:.3f} | "
                f"Win Rate: {recent_win_rate:.3f}"
            )

        if episode % save_interval == 0:
            save_path = f"models/dqn_episode_{episode}.pt"
            dqn_agent.save(save_path)
            print(f"Saved model to {save_path}")

    final_path = "models/dqn_final.pt"
    dqn_agent.save(final_path)
    print(f"Training finished. Final model saved to {final_path}")


# if __name__ == "__main__":
#     train(num_episodes=1000)
if __name__ == "__main__":
    train(
        num_episodes=5000,
        save_interval=1000,
    )