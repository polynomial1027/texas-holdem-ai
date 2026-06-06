import os
from statistics import mean

from src.holdem.game import create_default_game
from src.holdem.encoder import encode_observation
from src.agents.dqn_agent import DQNAgent
from src.agents.rule_based_agent import RuleBasedAgent


def train(
    num_episodes=3000,
    save_interval=500,
):
    """
    第二阶段训练：

    Player 1: DQNAgent
    Player 2: RuleBasedAgent

    目的：
    - 让 DQN 不再只适应随机玩家
    - 逼迫 DQN 学会面对更合理的对手
    - 降低空气牌乱 call / raise 的倾向
    """

    os.makedirs("models", exist_ok=True)

    game = create_default_game(num_players=2)

    dqn_agent = DQNAgent(
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.997,
    )

    rule_agent = RuleBasedAgent()

    episode_rewards = []
    episode_losses = []
    win_count = 0
    fold_count = 0
    call_count = 0
    raise_count = 0
    bet_count = 0
    check_count = 0

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

                if action == "fold":
                    fold_count += 1
                elif action == "call":
                    call_count += 1
                elif action == "raise":
                    raise_count += 1
                elif action == "bet":
                    bet_count += 1
                elif action == "check":
                    check_count += 1

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
                action = rule_agent.choose_action(observation)
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
            win_rate = win_count / episode

            total_actions = (
                fold_count
                + call_count
                + raise_count
                + bet_count
                + check_count
            )

            if total_actions > 0:
                fold_rate = fold_count / total_actions
                call_rate = call_count / total_actions
                raise_rate = raise_count / total_actions
                bet_rate = bet_count / total_actions
                check_rate = check_count / total_actions
            else:
                fold_rate = call_rate = raise_rate = bet_rate = check_rate = 0.0

            print(
                f"Episode {episode:5d} | "
                f"Avg Reward: {mean(recent_rewards):8.3f} | "
                f"Avg Loss: {mean(recent_losses):8.5f} | "
                f"Epsilon: {dqn_agent.epsilon:.3f} | "
                f"Win Rate: {win_rate:.3f} | "
                f"Actions F/C/R/B/X: "
                f"{fold_rate:.2f}/"
                f"{call_rate:.2f}/"
                f"{raise_rate:.2f}/"
                f"{bet_rate:.2f}/"
                f"{check_rate:.2f}"
            )

        if episode % save_interval == 0:
            save_path = f"models/dqn_vs_rule_episode_{episode}.pt"
            dqn_agent.save(save_path)
            print(f"Saved model to {save_path}")

    final_path = "models/dqn_final.pt"
    dqn_agent.save(final_path)
    print(f"Training finished. Final model saved to {final_path}")


if __name__ == "__main__":
    train(num_episodes=10000)