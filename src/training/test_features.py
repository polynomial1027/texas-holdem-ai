from src.holdem.cards import Card
from src.holdem.features import (
    extract_poker_feature_dict,
    encode_poker_features,
    get_poker_feature_dim,
)


def main():
    # K♥ Q♠ / 8♣ J♥ 5♦
    hole_cards = [
        Card(rank=13, suit="♥"),
        Card(rank=12, suit="♠"),
    ]

    board_cards = [
        Card(rank=8, suit="♣"),
        Card(rank=11, suit="♥"),
        Card(rank=5, suit="♦"),
    ]

    feature_dict = extract_poker_feature_dict(hole_cards, board_cards)
    feature_vec = encode_poker_features(hole_cards, board_cards)

    print("=" * 70)
    print("Poker Feature Test")
    print("=" * 70)

    for key, value in feature_dict.items():
        print(f"{key}: {value}")

    print("-" * 70)
    print("Feature vector shape:", feature_vec.shape)
    print("Expected dim:", get_poker_feature_dim())

    assert feature_vec.shape == (get_poker_feature_dim(),)

    print("=" * 70)
    print("Feature test passed.")


if __name__ == "__main__":
    main()