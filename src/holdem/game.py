from src.holdem.cards import Deck
from src.holdem.player import Player
from src.holdem.evaluator import evaluate_seven, hand_rank_name
from src.holdem.actions import Action


class TexasHoldemGame:
    """
    简化版两人德州扑克环境。

    当前支持：
    - 两名玩家
    - 筹码
    - 底池
    - fold / check / call / bet / raise
    - preflop / flop / turn / river / showdown
    - reset() / step(action)

    简化点：
    - 暂时不区分大小盲位置轮换
    - bet / raise 金额固定
    - 不做复杂 side pot
    """

    STREET_ORDER = ["preflop", "flop", "turn", "river", "showdown"]

    def __init__(
        self,
        players,
        small_blind: int = 5,
        big_blind: int = 10,
        fixed_bet: int = 20,
    ):
        if len(players) != 2:
            raise ValueError("This simplified environment currently supports exactly 2 players.")

        self.players = players
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.fixed_bet = fixed_bet

        self.deck = None
        self.board = []
        self.pot = 0

        self.street = "preflop"
        self.current_player_index = 0
        self.current_bet = 0
        self.last_raiser_index = None
        self.hand_over = False
        self.winners = []
        self.actions_this_street = 0
        self.starting_chips = [player.chips for player in self.players]

    def reset(self):
        self.deck = Deck()
        self.board = []
        self.pot = 0

        self.street = "preflop"
        self.current_player_index = 0
        self.current_bet = 0
        self.last_raiser_index = None
        self.hand_over = False
        self.winners = []
        self.actions_this_street = 0

        for player in self.players:
            player.reset_for_new_hand()

        # 记录本手开始前的筹码，用于计算更接近真实 chip delta 的 terminal reward。
        # 注意：这里必须在 posting blinds 之前记录，这样大小盲也会计入本手盈亏。
        self.starting_chips = [player.chips for player in self.players]

        self._post_blinds()
        self._deal_hole_cards()

        # heads-up 中，小盲位 preflop 先行动
        self.current_player_index = 0

        self._advance_until_action_available_or_hand_over()

        return self.get_observation()

    def _post_blinds(self):
        small_blind_player = self.players[0]
        big_blind_player = self.players[1]

        sb_paid = small_blind_player.bet(self.small_blind)
        bb_paid = big_blind_player.bet(self.big_blind)

        self.pot += sb_paid + bb_paid
        self.current_bet = self.big_blind
        self.last_raiser_index = 1

    def _deal_hole_cards(self):
        for player in self.players:
            player.receive_cards(self.deck.deal(2))

    def _deal_flop(self):
        self.board.extend(self.deck.deal(3))

    def _deal_turn(self):
        self.board.extend(self.deck.deal(1))

    def _deal_river(self):
        self.board.extend(self.deck.deal(1))

    def get_current_player(self):
        return self.players[self.current_player_index]

    def get_opponent(self):
        return self.players[1 - self.current_player_index]

    def get_legal_actions(self):
        player = self.get_current_player()

        if self.hand_over or player.folded or player.is_all_in():
            return []

        to_call = self.current_bet - player.current_bet

        legal_actions = []

        if to_call > 0:
            legal_actions.append(Action.FOLD)
            legal_actions.append(Action.CALL)

            if player.chips > to_call:
                legal_actions.append(Action.RAISE)
        else:
            legal_actions.append(Action.CHECK)

            if player.chips > 0:
                legal_actions.append(Action.BET)

        return legal_actions

    def step(self, action):
        """
        强化学习接口的核心。

        输入：
            action: Action 枚举，例如 Action.CALL

        返回：
            observation, reward, done, info
        """
        if self.hand_over:
            return self.get_observation(), 0, True, {"message": "Hand already over."}

        if isinstance(action, str):
            action = Action(action)

        legal_actions = self.get_legal_actions()

        if action not in legal_actions:
            raise ValueError(f"Illegal action {action}. Legal actions: {legal_actions}")

        player = self.get_current_player()
        opponent = self.get_opponent()

        reward = 0
        info = {
            "street": self.street,
            "player": player.name,
            "action": action.value,
        }

        self.actions_this_street += 1

        if action == Action.FOLD:
            player.fold()
            self.hand_over = True
            self.winners = [opponent]
            opponent.chips += self.pot
            info["result"] = f"{player.name} folded. {opponent.name} wins pot."

        elif action == Action.CHECK:
            self._advance_after_non_aggressive_action()

        elif action == Action.CALL:
            to_call = self.current_bet - player.current_bet
            paid = player.bet(to_call)
            self.pot += paid
            self._advance_after_non_aggressive_action()

        elif action == Action.BET:
            paid = player.bet(self.fixed_bet)
            self.pot += paid
            self.current_bet = player.current_bet
            self.last_raiser_index = self.current_player_index
            self._switch_player()

        elif action == Action.RAISE:
            to_call = self.current_bet - player.current_bet
            raise_amount = self.fixed_bet
            total_amount = to_call + raise_amount

            paid = player.bet(total_amount)
            self.pot += paid
            self.current_bet = player.current_bet
            self.last_raiser_index = self.current_player_index
            self._switch_player()

        if not self.hand_over and self._only_one_player_not_folded():
            self._finish_by_fold()

        if not self.hand_over and self.street == "showdown":
            self._showdown()

        if not self.hand_over:
            self._advance_until_action_available_or_hand_over()

        if self.hand_over:
            reward = self._calculate_terminal_reward_for_player(player)
            info["terminal_reward"] = reward

        done = self.hand_over

        return self.get_observation(), reward, done, info

    def _advance_after_non_aggressive_action(self):
        """
        check 或 call 之后判断是否结束当前下注轮。
        """
        if self._betting_round_complete():
            self._advance_street()
        else:
            self._switch_player()

    def _active_players(self):
        return [player for player in self.players if not player.folded]

    def _betting_round_complete(self):
        """
        判断当前下注轮是否已经结束。

        简化规则：
        - 已弃牌玩家不参与判断。
        - 非 all-in 玩家必须已经跟到当前最高下注。
        - 每个仍能行动的玩家至少需要行动过一次。
        """
        active_players = self._active_players()

        if len(active_players) <= 1:
            return True

        players_who_can_act = [
            player for player in active_players
            if not player.is_all_in()
        ]

        if not players_who_can_act:
            return True

        highest_bet = max(player.current_bet for player in active_players)

        for player in players_who_can_act:
            if player.current_bet < highest_bet:
                return False

        return self.actions_this_street >= len(players_who_can_act)

    def _all_active_players_all_in(self):
        active_players = self._active_players()
        return active_players and all(player.is_all_in() for player in active_players)

    def _finish_all_in_showdown(self):
        """
        当所有未弃牌玩家都已经 all-in 时，直接补齐公共牌并摊牌。
        """
        while self.street != "showdown":
            self._advance_street()

        self._showdown()

    def _advance_street(self):
        """
        进入下一阶段。
        """
        for player in self.players:
            player.current_bet = 0

        self.current_bet = 0
        self.last_raiser_index = None
        self.actions_this_street = 0

        if self.street == "preflop":
            self.street = "flop"
            self._deal_flop()
            self.current_player_index = 1

        elif self.street == "flop":
            self.street = "turn"
            self._deal_turn()
            self.current_player_index = 1

        elif self.street == "turn":
            self.street = "river"
            self._deal_river()
            self.current_player_index = 1

        elif self.street == "river":
            self.street = "showdown"

        elif self.street == "showdown":
            self._showdown()

    def _switch_player(self):
        self.current_player_index = 1 - self.current_player_index

    def _advance_until_action_available_or_hand_over(self):
        """
        如果当前玩家没有合法动作，例如已经 all-in，
        就自动推进到下一个玩家、下一条街，或直接摊牌。

        这个方法主要是为了避免强化学习训练时出现：
        No legal actions available.
        """
        safety_counter = 0

        while not self.hand_over:
            if self._all_active_players_all_in():
                self._finish_all_in_showdown()
                return

            legal_actions = self.get_legal_actions()

            if legal_actions:
                return

            if self._betting_round_complete():
                self._advance_street()
            else:
                self._switch_player()

            if self.street == "showdown":
                self._showdown()
                return

            safety_counter += 1

            if safety_counter > 20:
                debug_state = {
                    "street": self.street,
                    "current_player_index": self.current_player_index,
                    "current_bet": self.current_bet,
                    "pot": self.pot,
                    "players": [
                        {
                            "name": player.name,
                            "chips": player.chips,
                            "current_bet": player.current_bet,
                            "total_bet_this_hand": player.total_bet_this_hand,
                            "folded": player.folded,
                            "all_in": player.is_all_in(),
                        }
                        for player in self.players
                    ],
                }
                raise RuntimeError(
                    f"Failed to find available action. Debug state: {debug_state}"
                )

    def _only_one_player_not_folded(self):
        active_players = [player for player in self.players if not player.folded]
        return len(active_players) == 1

    def _finish_by_fold(self):
        active_player = [player for player in self.players if not player.folded][0]
        self.hand_over = True
        self.winners = [active_player]
        active_player.chips += self.pot

    def _showdown(self):
        results = []

        for player in self.players:
            if player.folded:
                continue

            seven_cards = player.hole_cards + self.board
            score, best_five = evaluate_seven(seven_cards)

            results.append({
                "player": player,
                "score": score,
                "rank_name": hand_rank_name(score),
                "best_five": best_five,
            })

        best_score = max(result["score"] for result in results)
        winner_results = [
            result for result in results
            if result["score"] == best_score
        ]

        self.winners = [result["player"] for result in winner_results]

        split_amount = self.pot // len(self.winners)

        for winner in self.winners:
            winner.chips += split_amount

        self.hand_over = True

    def _calculate_terminal_reward_for_player(self, player):
        """
        用 chip delta 计算 terminal reward。

        reward = 本手结束后玩家筹码 - 本手开始前玩家筹码

        这样 fold / call / raise / showdown / all-in 都统一由真实筹码变化决定，
        比旧版的 -player.total_bet_this_hand 更接近扑克训练目标。
        """
        player_index = self.players.index(player)
        return player.chips - self.starting_chips[player_index]

    def get_observation(self):
        """
        返回当前环境状态。

        后面接强化学习时，可以把这个 dict 转成 numpy array / tensor。
        """
        player = self.get_current_player()

        return {
            "street": self.street,
            "current_player": player.name,
            "hole_cards": player.hole_cards,
            "board": self.board,
            "pot": self.pot,
            "current_bet": self.current_bet,
            "player_chips": [p.chips for p in self.players],
            "player_current_bets": [p.current_bet for p in self.players],
            "player_folded": [p.folded for p in self.players],
            "legal_actions": [action.value for action in self.get_legal_actions()],
        }

    def render(self):
        print("=" * 70)
        print(f"Street: {self.street}")
        print(f"Board : {self.board}")
        print(f"Pot   : {self.pot}")
        print(f"Current bet: {self.current_bet}")
        print("-" * 70)

        for i, player in enumerate(self.players):
            marker = " <-- current" if i == self.current_player_index and not self.hand_over else ""
            print(f"{player.name}{marker}")
            print(f"  Hole cards: {player.hole_cards}")
            print(f"  Chips     : {player.chips}")
            print(f"  Current bet: {player.current_bet}")
            print(f"  Total bet : {player.total_bet_this_hand}")
            print(f"  Folded    : {player.folded}")

        if self.hand_over:
            print("-" * 70)
            print("Hand over.")
            print("Winner(s):", [player.name for player in self.winners])

        print("=" * 70)


def create_default_game(num_players=2):
    players = [Player(f"Player {i + 1}") for i in range(num_players)]
    return TexasHoldemGame(players)