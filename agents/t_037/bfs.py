from template import Agent
from Splendor.splendor_model import *
import random
import time
from copy import deepcopy

TIME_LIMITATION = 0.90
NUM_OF_AGENT = 2

CARD_SCORE = {
    5:['4b','4r','4B','4w','6B','7r3B','6b','6g','6r','7w3b','7b3g','4g'],
    4:['3g','2g1r','3B','1w2B','2w1b','3r','3w','2b1g','5w','7r','5b','5g','5B','7w','7b','3b','2r1B'],
    3:['2b2B','1r1b1B1g','2g2B','1r1w1B1g','2b2r','1r1w1B1b','2w2r','1g1w1B1b','2w2g','1g1w1r1b','5r3B','4r2B1g','5w3b','1r4B2w','5b3g','2b1B4w','3w5B','4b2g1w','5g3r','4g2r1b','3r6B3w','3b3B6w','6b3g3w','6g3r3b','6r3B3g']
}

SCORE_CARD = {'4b': 5, '4r': 5, '4B': 5, '4w': 5, '6B': 5, '7r3B': 5, '6b': 5, '6g': 5, '6r': 5, '7w3b': 5, '7b3g': 5, '4g': 5, '3g': 4, '2g1r': 4, '3B': 4, '1w2B': 4, '2w1b': 4, '3r': 4, '3w': 4, '2b1g': 4, '5w': 4, '7r': 4, '5b': 4, '5g': 4, '5B': 4, '7w': 4, '7b': 4, '3b': 4, '2r1B': 4, '2b2B': 3, '1r1b1B1g': 3, '2g2B': 3, '1r1w1B1g': 3, '2b2r': 3, '1r1w1B1b': 3, '2w2r': 3, '1g1w1B1b': 3, '2w2g': 3, '1g1w1r1b': 3, '5r3B': 3, '4r2B1g': 3, '5w3b': 3, '1r4B2w': 3, '5b3g': 3, '2b1B4w': 3, '3w5B': 3, '4b2g1w': 3, '5g3r': 3, '4g2r1b': 3, '3r6B3w': 3, '3b3B6w': 3, '6b3g3w': 3, '6g3r3b': 3, '6r3B3g': 3}



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
    

class myAgent(Agent):
    def __init__(self,_id):
        super().__init__(_id)
        self.game_rule = SplendorGameRule(NUM_OF_AGENT)
    
    def get_next_actions(self,actions,game_state,total_gems):
        
        next_actions = self.game_rule.getLegalActions(game_state, self.id)


        # return buy first
        buy_actions = [a for a in next_actions if "buy" in a["type"]]
        if buy_actions:
            return buy_actions, "buy"
    
        if total_gems < 8 and sum(game_state.board.gems.values()) > 2:
            collect= [a for a in next_actions if "collect" in a["type"]]
            if collect:
                return collect, "collect"
            
        reserve_actions = [a for a in next_actions if "reserve" in a["type"]]
        
        if reserve_actions:
            return reserve_actions, "reserve"
        
        if total_gems < 9 and sum(game_state.board.gems.values()) > 1:
            collect= [a for a in next_actions if "collect" in a["type"]]
            if collect:
                return collect, "collect"

        return random.choice(actions), "nothing_to_do"
    
    def buy(self, actions, state, myQ, path):
        actions.sort(key=lambda x: x['card'].points + (SCORE_CARD.get(x['card'].code, 1) - 3) * 0.5, reverse=True)
        next_state = self.game_rule.generateSuccessor(deepcopy(state),actions[0], self.id)
        new_path = path + [actions[0]]
        myQ.push((next_state,new_path))
        return
        
    def calculate_gem_shortage(self,card,total_gems) -> dict:
        shortages = 0
        for color, required in card.cost.items():
            available = total_gems.get(color, 0)
            if required > available:
                shortages += (required - available)
        return shortages
    def reserve(self,actions,state:SplendorState,myQ, path,cards):
        n_card = sum(len(c) for c in cards.values())
        best_action = None
        for a in actions:
            if best_action:
                if best_action["card"].points == 5 and n_card > 6:
                    next_state = self.game_rule.generateSuccessor(deepcopy(state),best_action,self.id)
                    new_path = path + [best_action]
                    myQ.push((next_state,new_path))
                    return
            else:
                best_action = a
                continue
            card = a["card"]
            if card.code in CARD_SCORE[5]:
                bonus = 1.5
            # elif card.code in CARD_SCORE[4]:
            #     bonus = 1
            # elif card.code in CARD_SCORE[3]:
            #     bonus = 0.5
            else:
                bonus = 0
            if n_card > 3:
                if card.points >= 4:
                    if best_action:
                        if card.points > best_action["card"].points:
                            best_action = a
                            continue
            if bonus >= 1.5:
                if best_action:
                    if card.points >= best_action["card"].points:
                        best_action = a
                        continue

            # Extra game strategy to stop your opponent's success
            # total_gems = {color: oponent_agent.gems.get(color, 0) + len([c for c in cards[color]]) for color in COLOURS.values()}
            # shortage = self.calculate_gem_shortage(card,total_gems)
            # if shortage == 0 and :
            #     next_state = self.game_rule.generateSuccessor(deepcopy(state),a,self.id)
            #     new_path = path + [a]
            #     myQ.push((next_state,new_path))
            #     return


            
            # Extra game strategy to stop your opponent's success
            # total_gems = {color: oponent_agent.gems.get(color, 0) + len([c for c in cards[color]]) for color in COLOURS.values()}
            # shortage = self.calculate_gem_shortage(card,total_gems)
            # if shortage == 0 and :
            #     next_state = self.game_rule.generateSuccessor(deepcopy(state),a,self.id)
            #     new_path = path + [a]
            #     myQ.push((next_state,new_path))
            #     return
        if best_action:
            next_state = self.game_rule.generateSuccessor(deepcopy(state),best_action,self.id)
            new_path = path + [best_action]
            myQ.push((next_state,new_path))
        return
                    
    
    def wether_collect(self,action,state:SplendorState,cards,score):
        n_card = sum(len(c) for c in cards.values())
        if n_card < 4:
            display_cards = state.board.dealt[0] + state.board.dealt[1]
            for c in cards["yellow"]:
                if c.points == 1:
                    display_cards.append(c)
        elif score > 11:
            display_cards = state.board.dealt[1] + state.board.dealt[2] + cards["yellow"]
        else:
            display_cards = state.board.dealt[0] + state.board.dealt[1] + state.board.dealt[2] + cards["yellow"]
        
        shortage_gems = {c: 0 for c in COLOURS.values()}
        # weight = 1
        for card in display_cards:
            if not card:
                continue
            for color, num in card.cost.items():
                shortage_gems[color] += num
                shortage_gems[color] -= len(cards[color])
        # for card in cards["yellow"]:
        #     if card.points > 3 and n_card < 5:
        #         continue
        #     if card.points == 4 and score > 10:
        #         weight = 3
        #     elif card.points == 5 and score > 9:
        #         weight = 4
        #     else:
        #         weight = 2
        #     for color, num in card.cost.items():
        #         shortage_gems[color] += num * weight
        #         shortage_gems[color] -= len(cards[color])
        max_value = max(shortage_gems.values())
        most_needed_gems = [gem for gem, amount in shortage_gems.items() if amount == max_value]

        return any((gem in action['collected_gems'].keys() and gem not in action["returned_gems"].keys()) for gem in most_needed_gems)    
    

    def collect(self,actions,state,myQ,path,cards,score):
        for a in actions:
            if(self.wether_collect(a,state,cards,score)):
                next_state = self.game_rule.generateSuccessor(deepcopy(state),a,self.id)
                new_path = path + [a]
                myQ.push((next_state,new_path))
                

    
    def SelectAction(self, actions, game_state:SplendorState):
        start_time = time.time()
        myQ = Queue()
        myQ.push((game_state,[]))
        agent = game_state.agents[self.id]
        score = agent.score
        total_gems = sum(agent.gems.values())
        second_best = None
        third_best = None
        forth_best = None
        while not myQ.isEmpty() and time.time() - start_time < TIME_LIMITATION:
            current_state, path = myQ.pop()
            new_score = current_state.agents[self.id].score
            score_diff = new_score - score
            if score_diff > 1:
                third_best = path[0]
            if score_diff > 2:
                second_best = path[0]
            if new_score >=15 or score_diff > 3:
                # print("success")
                return path[0]
            new_actions, actionType = self.get_next_actions(actions,current_state,total_gems)
            if actionType == "buy":
                self.buy(new_actions,current_state,myQ,path)
            elif actionType =="reserve":
                cards = current_state.agents[self.id].cards
                self.reserve(new_actions,current_state,myQ,path,cards)
            elif actionType == "collect":
                cards = current_state.agents[self.id].cards
                self.collect(new_actions,current_state,myQ,path,cards,new_score)
            else:
                next_state = self.game_rule.generateSuccessor(deepcopy(current_state),new_actions,self.id)
                new_path = path + [new_actions]
                myQ.push((deepcopy(next_state),new_path))
        if second_best:
            # if myQ.isEmpty():
            #     print("no path to go:1")
            # else:
            #     print("timeout:1")
            return second_best
        if third_best:
            # if myQ.isEmpty():
            #     print("no path to go:1")
            # else:
            #     print("timeout:1")
            return third_best
        # if forth_best:
        #     # if myQ.isEmpty():
        #     #     print("no path to go:1")
        #     # else:
        #     #     print("timeout:1")
        #     return forth_best
        # if myQ.isEmpty():
        #     print("no path to go")
        # else:
        #     print("timeout")
        # if not path:
        #     print("choose random")
        return path[0] if path else random.choice(actions) 
            
    



  
