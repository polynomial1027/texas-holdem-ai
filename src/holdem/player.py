class Player:
    def __init__(self, name: str, agent=None, chips: int = 1000):
        self.name = name
        self.agent = agent
        self.initial_chips = chips
        self.chips = chips

        self.hole_cards = []
        self.folded = False
        self.current_bet = 0
        self.total_bet_this_hand = 0

    def reset_stack(self):
        self.chips = self.initial_chips

    def reset_for_new_hand(self):
        self.hole_cards = []
        self.folded = False
        self.current_bet = 0
        self.total_bet_this_hand = 0

    def receive_cards(self, cards):
        self.hole_cards.extend(cards)

    def bet(self, amount: int) -> int:
        """
        从玩家筹码中下注。
        如果筹码不足，则自动 all-in。
        返回实际下注金额。
        """
        actual_amount = min(amount, self.chips)
        self.chips -= actual_amount
        self.current_bet += actual_amount
        self.total_bet_this_hand += actual_amount
        return actual_amount

    def fold(self):
        self.folded = True

    def is_all_in(self) -> bool:
        return self.chips == 0

    def __repr__(self):
        return (
            f"Player(name={self.name}, chips={self.chips}, "
            f"cards={self.hole_cards}, folded={self.folded})"
        )