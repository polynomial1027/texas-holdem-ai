from src.holdem.game import create_default_game
from src.agents.random_agent import RandomAgent


def main():
    game = create_default_game(num_players=2)

    agents = {
        "Player 1": RandomAgent(),
        "Player 2": RandomAgent(),
    }

    observation = game.reset()
    done = False

    game.render()

    step_count = 0

    while not done:
        current_player_name = observation["current_player"]
        agent = agents[current_player_name]

        action = agent.choose_action(observation)

        print(f"\nStep {step_count}")
        print(f"{current_player_name} chooses: {action}")

        observation, reward, done, info = game.step(action)

        print("Reward:", reward)
        print("Info:", info)

        game.render()

        step_count += 1

        if step_count > 100:
            raise RuntimeError("Game did not terminate. Possible logic bug.")


if __name__ == "__main__":
    main()