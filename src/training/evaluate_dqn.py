from collections import Counter
from statistics import mean

from src.holdem.game import create_default_game
from src.holdem.encoder import encode_observation
from src.agents.dqn_agent import DQNAgent
from src.agents.random_agent import RandomAgent


def evaluate(
    model_path="models/dqn_final.pt",
    num_episodes=1000,
    render=False,
):
    """
    评估训练好的 DQN。

    Player 1: DQNAgent
    Player 2: RandomAgent

    注意：
    - 这里会把 DQN epsilon 设置为 0
    - 也就是完全使用网络预测，不再随机探索
    """

    game = create_default_game(num_players=2)

    dqn_agent = DQNAgent()
    dqn_agent.load(model_path)
    dqn_agent.epsilon = 0.0

    random_agent = RandomAgent()

    dqn_wins = 0
    random_wins = 0
    ties = 0

    rewards = []
    action_counter = Counter()

    for episode in range(1, num_episodes + 1):
        observation = game.reset()
        done = False

        total_reward_for_dqn = 0

        while not done:
            current_player = observation["current_player"]

            if current_player == "Player 1":
                state = encode_observation(observation)
                legal_actions = observation["legal_actions"]

                action, action_index = dqn_agent.choose_action(
                    state,
                    legal_actions,
                )

                action_counter[action] += 1

                observation, reward, done, info = game.step(action)
                total_reward_for_dqn += reward

            else:
                action = random_agent.choose_action(observation)
                observation, reward, done, info = game.step(action)

            if render:
                print(f"Episode {episode}")
                print("Current player:", current_player)
                print("Action:", action)
                print("Info:", info)
                game.render()

        winner_names = [player.name for player in game.winners]

        if len(winner_names) > 1:
            ties += 1
        elif winner_names[0] == "Player 1":
            dqn_wins += 1
        else:
            random_wins += 1

        rewards.append(total_reward_for_dqn)

    print("=" * 70)
    print("DQN Evaluation Result")
    print("=" * 70)
    print(f"Model path       : {model_path}")
    print(f"Episodes         : {num_episodes}")
    print(f"DQN wins         : {dqn_wins}")
    print(f"Random wins      : {random_wins}")
    print(f"Ties             : {ties}")
    print(f"DQN win rate     : {dqn_wins / num_episodes:.3f}")
    print(f"Random win rate  : {random_wins / num_episodes:.3f}")
    print(f"Tie rate         : {ties / num_episodes:.3f}")
    print(f"Average reward   : {mean(rewards):.3f}")
    print("=" * 70)
    print("DQN action distribution:")
    total_actions = sum(action_counter.values())

    for action in ["fold", "check", "call", "bet", "raise"]:
        count = action_counter[action]
        ratio = count / total_actions if total_actions > 0 else 0
        print(f"{action:>5s}: {count:6d} | {ratio:.3f}")

    print("=" * 70)


if __name__ == "__main__":
    evaluate(
        model_path="models/dqn_final.pt",
        num_episodes=1000,
        render=False,
    )