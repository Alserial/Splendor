from template import Agent
from Splendor.splendor_model import SplendorGameRule
from copy import deepcopy
from Splendor.splendor_utils import *
import time
import random

TREE_DEPTH = 2
NUM_AGENTS = 2
THINK_TIME = 0.95

RESERVE_THRESHOLD1 = 0.75
RESERVE_THRESHOLD2 = 0.55

# for buying cards
CARD_POINT_WEIGHT = 3
CARD_COLOUR_WEIGHT = 6
CARD_NOBLE_WEIGHT = 6

# weights for different aspects in evaluation
# TODO: machine learning to find optimal weights.
NOBLE_WEIGHT = 6.5
GEM_VALUE_WEIGHT = 1.0
YELLOW_WEIGHT = 1.5 # yellow gem weight
COLOUR_SATISFY_WEIGHT = 2.0 # number of cards satisfied 70%
SCORE_WEIGHT = 5.0



class myAgent(Agent):
    def __init__(self,_id):
        super().__init__(_id)
        self.gameRule = SplendorGameRule(NUM_AGENTS)
        self.agentId = _id
        self.opponentId = 1 - _id

    def SelectAction(self,actions,game_state):
        self.start_time = time.time()
        # current player make the action first
        action, _ = self.Minimax(game_state, float('-inf'), float('inf'), TREE_DEPTH, self.agentId)
        return action if action is not None else random.choice(actions)
    
    def Minimax(self, game_state, alpha, beta, depth, curr_player):
        # if reach to the end level or the game has ended
        if depth == 0 or self.gameRule.gameEnds() == True:
            return None, self.evaluate(game_state)
        
        # If it's our turn
        if curr_player == self.agentId:
            return self.agent_turn(game_state, alpha, beta, depth)
        else: # if it's opponent's turn
            return self.opponent_turn(game_state, alpha, beta, depth)
    
    def opponent_turn(self, curr_state, alpha, beta, depth):
        min_value = float('inf')
        best_action = None
        actions = self.getActions(curr_state, self.opponentId)
        for action in actions:
            # If reach the time limit
            if time.time() - self.start_time > THINK_TIME:
                print("Time reached!!")
                break
            temp_state = deepcopy(curr_state)
            # Make the action
            next_state = self.gameRule.generateSuccessor(temp_state, action, self.opponentId)
            _, evaluation = self.Minimax(next_state, alpha, beta, depth - 1, self.agentId)
            if evaluation < min_value:
                min_value = evaluation
                best_action = action
            beta = min(min_value, beta)
            if beta <= alpha:
                break
        return best_action, min_value
    
    def agent_turn(self, curr_state, alpha, beta, depth):
        max_value = float('-inf')
        best_action = None
        actions = self.getActions(curr_state, self.agentId)
        for action in actions:
            # If reach the time limit
            if time.time() - self.start_time > THINK_TIME:
                print("Time reached!!")
                break
            temp_state = deepcopy(curr_state)
            # Make the action
            next_state = self.gameRule.generateSuccessor(temp_state, action, self.agentId)
            _, evaluation = self.Minimax(next_state, alpha, beta, depth - 1, self.opponentId)
            # find the maximum pay off by comparing each evaluations
            if evaluation > max_value:
                max_value = evaluation
                best_action = action
            # alpha beta pruing
            alpha = max(alpha, max_value)
            if alpha >= beta:
                break
        # print("Minimax find an action!!")
        return best_action, max_value
    
    def getActions(self, game_state, playerId):
        actions = []
        curr_player, curr_board = game_state.agents[playerId], game_state.board

        # get all the actions
        all_actions = self.gameRule.getLegalActions(game_state, playerId)
        # get all the valuable cards for tier 2 and 3
        dealt = curr_board.dealt_list()
        valuable_tier2 = [card for card in dealt if card.deck_id == 2 and card.points != 1]
        valuable_tier3 = [card for card in dealt if card.deck_id == 3 and card.points != 3]

        """Check if there are any buy actions"""
        buy_actions = [action for action in all_actions if "buy" in action["type"]]
        if buy_actions:
            return buy_actions

        """See which colour is wanted the most on the board, collect gems respondingly"""
        # Avoid returning gems
        collect_actions = [action for action in all_actions if "collect" in action["type"]]
        if collect_actions:
            collected_actions = self.collect_tokens(curr_player, collect_actions, valuable_tier2 + valuable_tier3, curr_board)
            return [action for action, count in collected_actions]
        
        reserve_actions = [action for action in all_actions if action["type"] == "reserve"]
        if reserve_actions:
            return reserve_actions

        # if no actions available, could randomly choose an action
        if len(actions) == 0:
            actions.append(random.choice(all_actions))
        return actions
    
    def collect_tokens(self, player, collect_actions, target_cards, board):
        token_priority = {}
        for card in target_cards:
            for colour, cost in card.cost.items():
                if colour not in token_priority:
                    token_priority[colour] = 0
                token_priority[colour] += max(0, cost - player.gems.get(colour, 0) - len(player.cards[colour]))

        token_priority = dict(sorted(token_priority.items(), key=lambda item: item[1], reverse=True))  

        actions = []
        if sum(player.gems.values()) <= 7:
            for action in collect_actions:
                total = 0
                for colour, count in action["collected_gems"].items():
                    if colour in token_priority.keys():
                        total += count
                actions.append((action, total))
            actions = sorted(actions, key=lambda x: x[1], reverse=True)
            return actions
            # return actions[0][0]
        elif sum(player.gems.values()) >= 8:
            for action in collect_actions:
                if sum(action["returned_gems"].values()) == 0:
                    total = 0
                    for colour, count in action["collected_gems"].items():
                        if colour in token_priority.keys():
                            total += count
                    actions.append((action, total))
            if len(actions) != 0:
                actions = sorted(actions, key=lambda x: x[1], reverse=True)
                return actions
                # return actions[0][0]
        
        return actions
        # return None
    
    def calculate_noble_proximity(self, player, noble):
        noble_colours = noble[1]
        proximity_score = 0
        for colour, required_count in noble_colours.items():
            player_count = len(player.cards[colour])
            # if this colour is satisfied
            if player_count >= required_count:
                proximity_score += 1
            else:
                proximity_score += player_count / required_count
        return proximity_score
    
    def evaluate(self, gameState):
        # evaluate the difference of scores
        total_score = 0
        agent_score = self.gameRule.calScore(gameState, self.agentId)
        opponent_score = self.gameRule.calScore(gameState, self.opponentId)
        points_diff = (agent_score - opponent_score) * SCORE_WEIGHT

        self_agent = gameState.agents[self.agentId]
        opponent_agent = gameState.agents[self.opponentId]
        board = gameState.board

        # how many cards self/opponent agent has for each colour
        self_colours = {c:0 for c in COLOURS.values()}
        opponent_colours = {c:0 for c in COLOURS.values()}
        for colour, cards in self_agent.cards.items():
            self_colours[colour] += len(cards)
        for colour, cards in opponent_agent.cards.items():
            opponent_colours[colour] += len(cards)
        
        # evaluate the number of cards satisfied by 0.7 for each players
        self_count = 0
        opponent_count = 0
        for level in board.decks:
            for card in level:
                proximity1 = 0
                proximity2 = 0
                for colour, count in card.cost.items():
                    player_count = self_agent.gems[colour] + self_colours[colour]
                    if player_count >= count:
                        proximity1 += 1
                    else:
                        proximity1 += player_count / count
                    
                    opponent_count = opponent_agent.gems[colour] + opponent_colours[colour]
                    if opponent_count >= count:
                        proximity2 += 1
                    else:
                        proximity2 += opponent_count / count
                if proximity1 / len(card.cost.keys()) >= RESERVE_THRESHOLD1:
                    self_count += 1
                if proximity2 / len(card.cost.keys()) >= RESERVE_THRESHOLD1:
                    opponent_count += 1

        card_score = (self_count - opponent_count) * COLOUR_SATISFY_WEIGHT
        
        # Evaluate the gem and wildcard (gold token) count
        self_gem_value = sum(self_agent.gems.values()) - self_agent.gems.get("yellow", 0)
        opponent_gem_value = sum(opponent_agent.gems.values()) - opponent_agent.gems.get("yellow", 0)

        self_wildcard_value = self_agent.gems.get("yellow", 0)
        opponent_wildcard_value = opponent_agent.gems.get("yellow", 0)

        gem_score = (self_gem_value - opponent_gem_value) * GEM_VALUE_WEIGHT
        yellow_score = (self_wildcard_value - opponent_wildcard_value) * YELLOW_WEIGHT
        
        # Evaluate proximity to attracting nobles
        self_noble_value = sum([self.calculate_noble_proximity(self_agent, noble) for noble in board.nobles])
        opponent_noble_value = sum([self.calculate_noble_proximity(opponent_agent, noble) for noble in board.nobles])
        noble_score = (self_noble_value - opponent_noble_value) * NOBLE_WEIGHT

        total_score = points_diff + card_score + gem_score + yellow_score + noble_score

        return total_score