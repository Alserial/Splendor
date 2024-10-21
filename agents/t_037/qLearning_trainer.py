from template import Agent
import time, random, json
from Splendor.splendor_model import *
from copy import deepcopy
class Queue:
    "A container with a first-in-first-out (FIFO) queuing policy."

    def __init__(self):
        self.list = []

    def push(self, item):
        "Enqueue the 'item' into the queue"
        self.list.insert(0, item)

    def pop(self):
        """
          Dequeue the earliest enqueued item still in the queue. This
          operation removes the item from the queue.
        """
        return self.list.pop()

    def isEmpty(self):
        "Returns true if the queue is empty"
        return len(self.list) == 0


THINKTIME = 0.9
NUM_PLAYERS = 2
ALPHA = 0.1
GAMMA = 0.9
EPSILON = 0.4
MIN_EPSILON = 0.01
EPSILON_DECAY = 0.995
NEG_INF = -999999
WEIGHTS_FILE = 'agents/t_037/q_weights.json'
CARD_SCORE = {
    5:['4b','4r','4B','4w','6B','7r3B','6b','6g','6r','7w3b','7b3g','4g'],
    4:['3g','2g1r','3B','1w2B','2w1b','3r','3w','2b1g','5w','7r','5b','5g','5B','7w','7b','3b','2r1B'],
    3:['2b2B','1r1b1B1g','2g2B','1r1w1B1g','2b2r','1r1w1B1b','2w2r','1g1w1B1b','2w2g','1g1w1r1b','5r3B','4r2B1g','5w3b','1r4B2w','5b3g','2b1B4w','3w5B','4b2g1w','5g3r','4g2r1b','3r6B3w','3b3B6w','6b3g3w','6g3r3b','6r3B3g']
}
class myAgent(Agent):
    def __init__(self, _id):
        super().__init__(_id)
        self.game_rule = SplendorGameRule(NUM_PLAYERS)
        # Load weights from a JSON file or initialize if the file doesn't exist
        try:
            self.weights = json.load(open(WEIGHTS_FILE))['values']
        except FileNotFoundError:
            self.weights = {}


    def get_next_actions(self,game_state:SplendorState):
        
        next_actions = self.game_rule.getLegalActions(game_state, self.id)
        useful_actions = []
        cards = game_state.agents[self.id].cards

        # return buy first
        nobles = game_state.board.nobles
        noble_colour = []
        card_len = {}
        for colour in cards:
            card_len[colour] = len(cards[colour])

        for noble in nobles:
            for c in noble[1]:
                if card_len[c] < noble[1][c]:
                    noble_colour.append(c)


        for a in next_actions:
            if "buy" in a["type"]:
                if len(noble_colour) > 0:
                    if a["card"].points < 3 and (a["card"].colour not in noble_colour):
                        continue
                else:
                    if a["card"].points < 1:
                        continue
                useful_actions.append(a)



            elif "reserve" in a["type"]: 
                if len(noble_colour) > 0:
                    if a["card"].points < 3 and (a["card"].colour not in noble_colour):
                        continue
                else:
                    if a["card"].points < 2:
                        continue
                useful_actions.append(a)
        return useful_actions



    def BfsSearch(self, actions, game_state:SplendorState):
        start_time = time.time()
        myQ = Queue()
        myQ.push((game_state,[]))
        agent = game_state.agents[self.id]
        score = agent.score
        second_best = None
        third_best = None
        while not myQ.isEmpty() and time.time() - start_time < THINKTIME:
            current_state, path = myQ.pop()
            new_score = current_state.agents[self.id].score
            score_diff = new_score - score
            if score_diff > 1:
                third_best = path[0]
            if score_diff > 2:
                second_best = path[0]
            if new_score >=15 or score_diff > 3:
                print("excellent")
                # print(path)
                return path[0]
            for a in self.get_next_actions(current_state):
                next_state = self.game_rule.generateSuccessor(deepcopy(current_state),a,self.id)
                new_path = path + [a]
                myQ.push((next_state,new_path))
        if second_best:
            print("great")
            return second_best
        if third_best:
            print("soso")
            return third_best
        print("bad")
        return path[0] if path else random.choice(actions) 


    def SelectAction(self, actions, game_state:SplendorState):
        # self.weights = json.load(open(WEIGHTS_FILE))['values']
        start_time = time.time()
        n = random.random()
        best_action = random.choice(actions)
        opponent_id = 1 - self.id

        if game_state.agents[self.id].score >= 11:
            best_action = self.BfsSearch(actions,game_state)
            return best_action
        else:
            if len(actions) > 1:
                if n < EPSILON:
                    best_action = random.choice(actions)
                    best_q_value = self.GetQValue(game_state,best_action,self.id)
                else:
                    best_action,best_q_value = self.FindBest(actions, game_state, start_time, self.id)


                # update weights training
                # Execute the chosen action
                next_state = self.game_rule.generateSuccessor(deepcopy(game_state), best_action, self.id)
    
                # get opponent legal actions
                opponent_actions = self.game_rule.getLegalActions(next_state,opponent_id)
                best_opponent_action,_ = self.FindBest(opponent_actions,next_state,start_time,opponent_id)
                # Execute the chosen action
                next_state = self.game_rule.generateSuccessor(next_state, best_opponent_action, opponent_id)


                # Calculate the reward and update Q-values
                reward = self.GetReward(best_action, game_state)
                next_actions = self.game_rule.getLegalActions(next_state, self.id)
                _,next_best_q_value = self.FindBest(next_actions, next_state, start_time, self.id)
                self.UpdateWeights(game_state, best_action, reward, best_q_value, next_best_q_value, self.id)
        return best_action

    def GetQValue(self, state, action, _id):
        features = self.ExtractFeatures(state, action)
        return sum(features[key] * self.weights.get(key,0) for key in features)
    
    def normalize_dict_values(self,input_dict):
        if not input_dict:
            return input_dict
        
        max_value = max(input_dict.values())
        
        if max_value == 0:
            return {k: 0 for k in input_dict}
        
        normalized_dict = {k: v / max_value for k, v in input_dict.items()}
        return normalized_dict

    def can_buy_card(self,agent_cards, agent_gems, card_cost):
        count = 0
        for color, cost in card_cost.items():
            total_available = len(agent_cards.get(color, 0)) + agent_gems.get(color, 0) + len(agent_cards.get("yellow",0))
            if total_available < cost:
                count += cost - total_available
        return count

    def ExtractFeatures(self, state: SplendorState, action):
        features = {}

        board = state.board
        agent = state.agents[self.id]
        opponent = state.agents[1 - self.id]

        yellow_gems = agent.gems["yellow"]
        cards = agent.cards
        
        
        features["my_score"] = agent.score
        features["agent_score"] = opponent.score
        if sum(agent.gems.values()) > 8:
            features["total_gems"] = 0
        else:
            features["total_gems"] = 1
        


        # can_buy_count = 0
        # for i in range(3):
        #     for card in board.dealt[i]:
        #         if card is None:
        #             continue
        #         if all(card.cost[colour] <= len(agent.cards[colour]) + agent.gems[colour] for colour in card.cost):
        #             can_buy_count += 1
        # features["can_buy_count"] = can_buy_count / 10

        noble_potential = 0
        # red,black
        noble_colours = {}
        for noble in board.nobles:
            unmet_requirements = 0
            for colour in noble[1]:
                if colour in noble_colours:
                    noble_colours[colour] += 1
                else:
                    noble_colours[colour] = 1
                unmet_requirements += max(0, noble[1][colour] - len(agent.cards[colour]))
            noble_potential += 1 / (unmet_requirements + 1)
        features["noble_potential"] = noble_potential / 3


        # choose banned colour
        banned_colour = None
        if noble_colours.get("green",0) == 0:
            banned_colour = "green"
        elif noble_colours.get("blue",0) == 0:
            banned_colour = "blue"
        elif noble_colours.get("black",0) == 0:
            banned_colour = "black"
        elif noble_colours.get("red",0) == 0:
            banned_colour = "red"
        elif noble_colours.get("white",0) == 0:
            banned_colour = "white"
        if not banned_colour:
            if noble_colours.get("green",0) == 1:
                banned_colour = "green"
            elif noble_colours.get("blue",0) == 1:
                banned_colour = "blue"
            elif noble_colours.get("black",0) == 1:
                banned_colour = "black"
            elif noble_colours.get("red",0) == 1:
                banned_colour = "red"
            elif noble_colours.get("white",0) == 1:
                banned_colour = "white"


        # check oberserve scope
        card_num = {"black": 0, "red": 0, "green": 0, "blue": 0, "white": 0}
        for c in agent.cards.keys():
            if c in card_num:
                card_num[c] += len(agent.cards[c])
        n_card = sum(card_num.values())
        if n_card < 3:
            dealt_ratio = [1,0,0]
        if n_card < 5:
            dealt_ratio = [1,0.2,0]
            phase = 1
        elif max(card_num.values()) > 5:
            dealt_ratio = [1,1,1]
            phase = 4
        elif max(card_num.values()) > 4:
            dealt_ratio = [1,1,0.5]
            phase = 3
        else:
            dealt_ratio = [1,1,0.1]
            phase = 2

        # get observe cards depand on phase
        total_gem_demand = {"black": 0, "red": 0, "green": 0, "blue": 0, "white": 0}
        for i in range(3):
            for card in board.dealt[i]:
                if not card:
                    continue
                for colour in card.cost:
                    diff = card.cost[colour] - len(agent.cards[colour])
                    if diff > 0:
                        total_gem_demand[colour] += (1*dealt_ratio[i])
        
        total_gem_demand = self.normalize_dict_values(total_gem_demand)
        
        if "collect" in action["type"]:
            for i in range(3):
                gem_demand = {"black": 0, "red": 0, "green": 0, "blue": 0, "white": 0, "yellow": 0}
                for card in board.dealt[i]:
                    if card is None:
                        continue
                    if card.colour == banned_colour and len(cards[banned_colour]) >= 2:
                        punish = 0.3
                    else:
                        punish = 1
                    for colour in card.cost:
                        if card.cost[colour] > len(agent.cards[colour]) + agent.gems[colour]:
                            if total_gem_demand[card.colour] <= 0.01:
                                gem_demand[colour] += (1*punish*0.01)
                            else:
                                gem_demand[colour] += (1*punish*total_gem_demand[card.colour])
                right = 0
                for colour in action["collected_gems"]:
                    if action["collected_gems"][colour] > 1:
                        right += 1.5 * gem_demand[colour]
                    else:
                        right += gem_demand[colour]
                for colour in action["returned_gems"]:
                    right -= action["returned_gems"][colour] * gem_demand[colour]
                features[f"dealt{i}_gem_demand"] = (right / 100) * dealt_ratio[i]

            # yellow_demand = sum(gem_demand.values())
            # yellow_usage_potential = yellow_demand * yellow_gems
            # features["yellow_gem_utility"] = yellow_usage_potential / 100
        else:
            for i in range(3):
                features[f"dealt{i}_gem_demand"] = 0
            # features["yellow_gem_utility"] = 0





        if "buy" in action["type"]:
            card = action['card']
            if min([len(c) for k, c in cards.items() if k != "yellow"]) < 2:
                features["card_points"] = card.points / 15
            else:
                features["card_points"] = card.points / 5

            oppo_diff = self.can_buy_card(opponent.cards,opponent.gems,card.cost)

            
            if oppo_diff > 8:
                features["oppo_buy"] = 0
            else:
                features["oppo_buy"] = (8-oppo_diff) / 8
            if card.points + agent.score >= 15:
                features["buy_win"] = card.points
            else:
                features["buy_win"] = 0
            
            count = 0
            for colour in card.cost:
                cost = card.cost[colour] - len(cards[colour])
                if cost > 0:
                    count += cost
            if n_card < 4:
                features["card_cost"] = count / 10
            else:
                features["card_cost"] = count / 3
            if min([len(c) for k, c in cards.items() if k != "yellow"]) < 2 or len(board.nobles) == 0:
                features["noble_demand"] = 0
            else:
                noble_demand = 0
                for noble in board.nobles:
                    if card.colour in noble[1] and (len(cards[card.colour]) < noble[1][card.colour]):
                        noble_demand += 1
                features["noble_demand"] = noble_demand / 3
            if card.code in CARD_SCORE[5]:
                features["card_rating"] = 1
            elif card.code in CARD_SCORE[4]:
                features["card_rating"] = 0.5
            else:
                features["card_rating"] = 0
            if card.colour == banned_colour and len(cards[banned_colour]) >= 2:
                features["card_rating"] -= 0.8
            if n_card < 5:
                features["colour_importance"] = total_gem_demand[card.colour]*2
            else:
                features["colour_importance"] = total_gem_demand[card.colour]
        else:
            features["card_points"] = 0
            features["card_cost"] = 0
            features["oppo_buy"] = 0
            features["noble_demand"] = 0
            features["card_rating"] = 0
            features["colour_importance"] = 0

        if ("noble" in action) and (action["noble"] is not None):
            features["get_noble"] = 1
        else:
            features["get_noble"] = 0

        if "reserve" in action["type"]:
            feat = 0
            oppo_actions = self.game_rule.getLegalActions(state, 1 - self.id)
            for oppo_action in oppo_actions:
                if "buy" in oppo_action["type"]:
                    card = oppo_action["card"]
                    # if card.points == 5:
                    #     feat = 1
                    #     break
                    # elif card.points == 4:
                    #     feat = 0.5
                    #     break
                    if ("noble" in oppo_action) and (oppo_action["noble"] is not None):
                        noble = oppo_action["noble"]
                        if (card.colour in noble[1]) and (len(opponent.cards[card.colour]) + 1 == noble[1][card.colour]):
                            feat += 1
                        if feat > 1:
                            break
            if feat == 1:
                features["defence"] = 1
            else:
                features["defence"] = 0

            card = action["card"]
            if card.colour == banned_colour:
                features["reserve_rating"] = 1
            if card.points == 5:
                features["reserve_rating"] = 0
            elif card.points == 4 or card.code in CARD_SCORE[5]:
                features["reserve_rating"] = 0.5
            else:
                features["reserve_rating"] = 1
            if feat == 1:
                features["reserve_rating"] = 0
        else:
            features["defence"] = 0
            features["reserve_rating"] = 0

        # oppo_defence = 0
        # for colour in opponent.cards:
        #     oppo_defence += len(opponent.cards[colour])
        # features["oppo_defence"] = oppo_defence / 10

        return features



    def find_new_cards(self,my_cards, my_next_cards):
        for colour in COLOURS.values():
            set_current_cards = set(my_cards[colour])
            set_next_cards = set(my_next_cards[colour])
            new_card_set = set_next_cards - set_current_cards

            if new_card_set:
                return (colour,new_card_set)
        return None        

    def FindBest(self, actions, state, start_time, _id):
        best_action = random.choice(actions)
        best_q_value = NEG_INF
        for action in actions:
            if time.time() - start_time > THINKTIME:
                break
            q_value = self.GetQValue(state, action, _id)
            # if "buy" in action["type"]:
            #     print("buy", q_value)
            # elif "collect" in action["type"]:
            #     print("collect", q_value)
            if q_value > best_q_value:
                best_q_value = q_value
                best_action = action
        return best_action, best_q_value

    # def GetReward(self, state:SplendorState, next_state:SplendorState, id):
    #     # difference in points
    #     score = next_state.agents[id].score - state.agents[id].score
    #     # score -= (oponent_next_state.agents[oponent_id].score - state.agents[oponent_id].score)

    #     # wether buy card
    #     my_cards = state.agents[id].cards
    #     my_next_cards = next_state.agents[id].cards

    #     # o_cards = state.agents[oponent_id].cards
    #     # o_next_cards = oponent_next_state.agents[oponent_id].cards
    #     different_card = self.find_new_cards(my_cards,my_next_cards)
    #     if different_card:
    #         for card[1] in different_card:
    #             card = set(my_next_cards) - set(my_cards)
    #             if card.code in CARD_SCORE[5]:
    #                 score += 1.5
    #             elif card.code in CARD_SCORE[4]:
    #                 score += 1
    #             elif card.code in CARD_SCORE[3]:
    #                 score += 0.5
    #             else:
    #                 if card.points < 3:
    #                     score -= 2
    #             score += card.points
    #             if different_card[0] == "yellow":
    #                 score -= 2
        
    #     # if len(o_cards) != len(o_next_cards):
    #     #     card = set(o_next_cards) - set(o_cards)
    #     #     if card.code in CARD_SCORE[5]:
    #     #         score -= 1.5
    #     #     elif card.code in CARD_SCORE[4]:
    #     #         score -= 1
    #     #     elif card.code in CARD_SCORE[3]:
    #     #         score -= 0.5
    #     #     else:
    #     #         if card.points < 3:
    #     #             score += 0.5

    #     return score

    def GetReward(self,action,state:SplendorState):
        score = 0
        agent = state.agents[self.id]
        # return buy first
        if "buy" in action["type"]:
            card = action["card"]
            if card.code in CARD_SCORE[5]:
                score += 1.5
            elif card.code in CARD_SCORE[4]:
                score += 1
            elif card.code in CARD_SCORE[3]:
                score += 0.5
            else:
                if card.points < 3:
                    score -= 1.5
            score += card.points
            
        elif "reserve" in action["type"]:
            card = action["card"]
            if card.code in CARD_SCORE[5]:
                score += 1
            elif card.code in CARD_SCORE[4]:
                score += 0
            elif card.code in CARD_SCORE[3]:
                score -= 1
            else:
                if card.points < 3:
                    score -= 3
            if card.points == 5:
                score += 1
            if agent.score > 8:
                score += card.points
        return score




    def UpdateWeights(self, state, action, reward, current_q_value, next_q_value, _id):
        temp = {}
        features = self.ExtractFeatures(state, action)

        if not hasattr(self, 'weights'):
            self.weights = {}
        for key in features.keys():
            if key not in self.weights:
                self.weights[key] = 0.1

        for key in features:
            self.weights[key] += ALPHA * (reward + GAMMA * next_q_value - current_q_value) * features[key]

        temp['values'] = self.weights
        with open(WEIGHTS_FILE, 'w') as output:
            json.dump(temp, output)


