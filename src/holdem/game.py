from src.holdem.cards import Deck
from src.holdem.player import Player
from src.holdem.evaluator import evaluate_seven, hand_rank_name


class TexasHoldemGame:
    def __init__(self, players):
        if len(players) < 2:
            raise ValueError("Texas Hold'em requires at least 2 players.")
        self.players = players
        self.deck = None
        self.board = []

    def reset(self):
        self.deck = Deck()
        self.board = []

        for player in self.players:
            player.reset_for_new_hand()

    def deal_hole_cards(self):
        for player in self.players:
            player.receive_cards(self.deck.deal(2))

    def deal_board(self):
        # 简化版：一次性发出5张公共牌
        self.board = self.deck.deal(5)

    def showdown(self):
        results = []

        for player in self.players:
            seven_cards = player.hole_cards + self.board
            score, best_five = evaluate_seven(seven_cards)

            results.append({
                "player": player,
                "score": score,
                "rank_name": hand_rank_name(score),
                "best_five": best_five,
            })

        best_score = max(result["score"] for result in results)
        winners = [result for result in results if result["score"] == best_score]

        return results, winners

    def play_one_hand(self, verbose=True):
        self.reset()
        self.deal_hole_cards()
        self.deal_board()

        results, winners = self.showdown()

        if verbose:
            print("=" * 60)
            print("Board:", self.board)
            print("-" * 60)

            for result in results:
                player = result["player"]
                print(f"{player.name}")
                print(f"  Hole Cards: {player.hole_cards}")
                print(f"  Best Hand : {result['best_five']}")
                print(f"  Rank      : {result['rank_name']}")
                print(f"  Score     : {result['score']}")
                print()

            print("-" * 60)

            if len(winners) == 1:
                print("Winner:", winners[0]["player"].name)
            else:
                print("Tie:", [winner["player"].name for winner in winners])

            print("=" * 60)

        return results, winners


def create_default_game(num_players=2):
    players = [Player(f"Player {i + 1}") for i in range(num_players)]
    return TexasHoldemGame(players)