from src.holdem.game import create_default_game


def main():
    game = create_default_game(num_players=2)

    for i in range(5):
        print(f"\nHand {i + 1}")
        game.play_one_hand(verbose=True)


if __name__ == "__main__":
    main()