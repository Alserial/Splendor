from template import Agent
import random
import time
from math import sqrt, log
from copy import deepcopy
from Splendor.splendor_model import *

TIME_LIMIT = 0.9
NUM_OF_AGENT = 2

class myAgent(Agent):
    def __init__(self, _id):
        super().__init__(_id)
        self.game_rule = SplendorGameRule(NUM_OF_AGENT)

    def SelectAction(self, actions, game_state: SplendorState):
        Start_time = time.time()
        c = sqrt(2)
        root = Node(game_state,self,None,None)
        current_node = root
        depth = 0
        while(time.time()-Start_time) < TIME_LIMIT:
            print('depth',depth)
            print(time.time() - Start_time)
            if not current_node.is_fully_expanded():
                current_node = current_node.expand()
                simulation_result = current_node.simulate(Start_time)
                print('simulation_result',simulation_result)
                current_node.update(simulation_result)
                current_node = current_node.parent
            elif current_node.is_fully_expanded() and current_node.children and depth <2:
                current_node = current_node.best_child(c)
                depth +=1
            else:
                break

        best_action = max(root.children, key=lambda x: x.visits).action if root.children else None

        return best_action if best_action else random.choice(actions)


def getAction(game_state, game_rule,agent_id):
    if agent_id:
        oppo_id = 0
    else:
        oppo_id = 1
    select_action=[]
    total_gems = sum(game_state.agents[agent_id].gems.values())
    all_actions = game_rule.getLegalActions(game_state, agent_id)
    buy_actions = sorted((action for action in all_actions if "buy" in action["type"]),
                         key=lambda x: x['card'].points, reverse = True)
    if buy_actions:
        add_num = min(len(buy_actions),3)
        select_action.extend(buy_actions[:add_num])
    
    # never have more than 10 tokens   
    if total_gems <=8:
        gem_actions= [action for action in all_actions if "collect" in action["type"]]
        if gem_actions:
            select_action.extend(gem_actions)
    
    # reserve oppo_can buy
    reserve_oppo = []
    #get oppo gem value
    oppo_gem = game_state.agents[oppo_id].gems
    #check delta 
    for card in game_state.board.dealt_list():
        if card.points >= 3:
            if can_buy(card,oppo_gem):
                reserve_oppo.append(card)
    reserve_action_all = [action for action in all_actions if "reserve" in action["type"]]
    reserve_oppo = [action for action in reserve_action_all if action['card'] in reserve_oppo]
    if reserve_oppo:
        select_action.extend(reserve_oppo)
    reserve_actions_high = [action for action in reserve_action_all if action['card'].points >=4 and 
                            action not in reserve_oppo]
    if reserve_actions_high:
        select_action.extend(reserve_actions_high)
    if(select_action):
        print('actionlength',len(select_action))
        return select_action

    return [random.choice(all_actions)]

def can_buy(card, gems):
    need=card.cost
    yellow_have = gems.get('yellow',0)
    for gem, value in need.items():
        if value > gems[gem]:
            yellow_needed = value - gems[gem]
            if yellow_have >= yellow_needed:
                yellow_have -= yellow_needed
            else:
                return False

    return True
class Node:
    def __init__(self, state, agent, parent, action):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.wins = 0
        self.visits = 0
        self.agent = agent
        self.untried_actions = agent.game_rule.getLegalActions(state, agent.id)
           
    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def expand(self):
        action = self.untried_actions.pop()
        next_state = self.agent.game_rule.generateSuccessor(deepcopy(self.state), action, self.agent.id)
        child_node = Node(next_state,self.agent, parent=self, action=action)
        self.children.append(child_node)
        return child_node

    def best_child(self, c_param):
        #choices_weights = [(child.wins / child.visits) + c_param * sqrt((2 * log(self.visits) / child.visits)) for child in self.children]
        choices_weights = [(child.wins/child.visits for child in self.children)]
        return self.children[choices_weights.index(max(choices_weights))]

    
    def update(self, reward):
        self.visits += 1
        if reward:
            self.wins += 1

    
    def simulate(self,Start_time):
        current_simulation_state = deepcopy(self.state)
        if self.agent.id:
            oppo_id = 0
        else:
            oppo_id = 1
        depth = 0
        while depth <20 and Start_time - time.time() < TIME_LIMIT:
            print('simulating')
            self_possible_actions = getAction(current_simulation_state,self.agent.game_rule,self.agent.id)
            oppo_possible_actions = getAction(current_simulation_state,self.agent.game_rule,oppo_id)
            self_action = random.choice(self_possible_actions) if self_possible_actions else None
            oppo_action = random.choice(oppo_possible_actions) if oppo_possible_actions else None
            current_simulation_state = self.agent.game_rule.generateSuccessor(current_simulation_state, oppo_action, oppo_id)
            current_simulation_state = self.agent.game_rule.generateSuccessor(current_simulation_state, self_action, self.agent.id)
            depth +=1

        return self.agent.game_rule.calScore(current_simulation_state, self.agent.id) > self.agent.game_rule.calScore(current_simulation_state, oppo_id)

