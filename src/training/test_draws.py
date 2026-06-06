from src.holdem.cards import Card
from src.holdem.draws import analyze_draws, encode_draw_features, get_draw_feature_dim


def print_case(title, hole_cards, board_cards):
    print("=" * 80)
    print(title)
    print("=" * 80)
    print("Hole:", hole_cards)
    print("Board:", board_cards)

    info = analyze_draws(hole_cards, board_cards)
    vec = encode_draw_features(hole_cards, board_cards)

    for key, value in info.items():
        print(f"{key}: {value}")

    print("-" * 80)
    print("Draw feature shape:", vec.shape)
    print("Expected dim:", get_draw_feature_dim())

    assert vec.shape == (get_draw_feature_dim(),)


def main():
    # Air hand: 3♦ 2♠ / A♣ K♦ Q♥
    print_case(
        "Air hand facing danger board",
        hole_cards=[
            Card(rank=3, suit="♦"),
            Card(rank=2, suit="♠"),
        ],
        board_cards=[
            Card(rank=14, suit="♣"),
            Card(rank=13, suit="♦"),
            Card(rank=12, suit="♥"),
        ],
    )

    # Low flush draw: 2♦ 3♦ / A♦ K♦ Q♥
    print_case(
        "Low flush draw",
        hole_cards=[
            Card(rank=2, suit="♦"),
            Card(rank=3, suit="♦"),
        ],
        board_cards=[
            Card(rank=14, suit="♦"),
            Card(rank=13, suit="♦"),
            Card(rank=12, suit="♥"),
        ],
    )

    # Open-ended straight draw: J♠ T♠ / 9♦ 8♣ 2♥
    print_case(
        "Open-ended straight draw",
        hole_cards=[
            Card(rank=11, suit="♠"),
            Card(rank=10, suit="♠"),
        ],
        board_cards=[
            Card(rank=9, suit="♦"),
            Card(rank=8, suit="♣"),
            Card(rank=2, suit="♥"),
        ],
    )

    print("=" * 80)
    print("Draw tests passed.")


if __name__ == "__main__":
    main()