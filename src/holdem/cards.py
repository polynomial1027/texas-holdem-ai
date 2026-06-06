from dataclasses import dataclass
import random


SUITS = ["♠", "♥", "♦", "♣"]
RANKS = list(range(2, 15))  # 2-10, J=11, Q=12, K=13, A=14


RANK_NAME = {
    2: "2",
    3: "3",
    4: "4",
    5: "5",
    6: "6",
    7: "7",
    8: "8",
    9: "9",
    10: "T",
    11: "J",
    12: "Q",
    13: "K",
    14: "A",
}


@dataclass(frozen=True)
class Card:
    rank: int
    suit: str

    def __str__(self) -> str:
        return f"{RANK_NAME[self.rank]}{self.suit}"

    def __repr__(self) -> str:
        return str(self)


class Deck:
    def __init__(self):
        self.cards = [Card(rank, suit) for suit in SUITS for rank in RANKS]
        self.shuffle()

    def shuffle(self) -> None:
        random.shuffle(self.cards)

    def deal(self, n: int = 1):
        if n > len(self.cards):
            raise ValueError("Not enough cards left in the deck.")
        dealt = self.cards[:n]
        self.cards = self.cards[n:]
        return dealt