class Player:
    def __init__(self, name: str, agent=None):
        self.name = name
        self.agent = agent
        self.hole_cards = []
        self.chips = 1000
        self.folded = False

    def reset_for_new_hand(self):
        self.hole_cards = []
        self.folded = False

    def receive_cards(self, cards):
        self.hole_cards.extend(cards)

    def __repr__(self):
        return f"Player(name={self.name}, cards={self.hole_cards})"