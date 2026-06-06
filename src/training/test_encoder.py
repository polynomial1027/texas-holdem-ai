from src.holdem.game import create_default_game
from src.holdem.encoder import encode_observation, get_state_dim, get_action_dim


def main():
    game = create_default_game(num_players=2)

    observation = game.reset()
    state = encode_observation(observation)

    print("=" * 70)
    print("Raw observation:")
    print(observation)

    print("=" * 70)
    print("Encoded state:")
    print(state)

    print("=" * 70)
    print("State shape:", state.shape)
    print("Expected state dim:", get_state_dim())
    print("Action dim:", get_action_dim())

    assert state.shape == (get_state_dim(),)

    print("=" * 70)
    print("Encoder test passed.")


if __name__ == "__main__":
    main()