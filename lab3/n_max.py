# lets try assingning eachc activity valuse according to A and B in a pair (x,y) 
# where x is the score by A on an actiivity and B is the score by B on the same activity. 
# The scores are based on how much they enjoy or prefer that activity.

TOTAL_TIME = 24 # ON SUNDAY I HAVE TOTAL 24 HRS
MAX_PLAYERS =10
activities = {

    "Coding": {
        "duration": 2,
        "utility": [9, 4, 7, 6, 8, 5, 7, 9, 4, 6]
    },

    "Exam Prep": {
        "duration": 2,
        "utility": [8, 6, 7, 5, 9, 4, 6, 8, 5, 7]
    },

    "Research": {
        "duration": 3,
        "utility": [8, 5, 6, 7, 5, 8, 9, 6, 7, 4]
    },

    "Sleep": {
        "duration": 7,
        "utility": [3, 9, 8, 5, 6, 7, 4, 8, 9, 7]
    },

    "Cooking": {
        "duration": 2,
        "utility": [4, 6, 5, 8, 7, 5, 9, 6, 4, 7]
    },

    "Eat Outside": {
        "duration": 2,
        "utility": [5, 8, 7, 9, 6, 8, 5, 7, 9, 6]
    },

    "Movie": {
        "duration": 2,
        "utility": [5, 8, 7, 6, 9, 7, 8, 5, 6, 9]
    },

    "Games": {
        "duration": 2,
        "utility": [6, 7, 5, 9, 8, 6, 7, 9, 5, 8]
    },

    "Reading": {
        "duration": 2,
        "utility": [6, 3, 4, 7, 5, 9, 8, 6, 7, 5]
    },

}

  

# the state of the game should be like this 
# 0 - A
# 1 - B
# 2 - C
# state = {
#     "remaining_time": TOTAL_TIME,
#     "selected_activities": [],
#     "turn" : 0
# }


def getLegalActions(state):
    actions = []
    for activity in activities : 
        duration = activities[activity]["duration"]
        
        # the activities are leagl if they are not in teh selected acticities and they can be done in remaining time 
        if duration <= state["remaining_time"] and activity not in state["selected_activities"]:
            actions.append(activity)
    return actions 



def isTerminal(state) :      
    return True if len(getLegalActions(state)) == 0 else False
  
  
def terminalUtility(state) : 
    n = state["players"]
    # utility vector which needs to be returned at the end 
    utiliies = [0] * n   
 
    leagalActivities = state["selected_activities"]

    for activity in leagalActivities: 
        activity_utilities = activities[activity]["utility"]
        
        for player in range(n): 
            utiliies[player] += activity_utilities[player]
            
    return utiliies

# makeMove() creates the new game state after the current player
    # chooses one specific activity.
    #
    # It performs three main state updates:
    # 1. Subtracts the chosen activity's duration from the remaining time.
    # 2. Adds the chosen activity to the list of selected activities.
    # 3. Switches the turn to the other player.
    #
    # The original state is not modified. Instead, a new state is returned.
    # This is important because Minimax will use the current state to create
    # multiple different child states for different possible activities.

def makeMove(state, activity ) :

    # right now i have 3 players  
    # Since we currently have only three players:
    # A -> B
    # B -> C
    # C -> A
    n = state["players"]
    
    def findNextTurn(X): 
        return (X + 1) % n

    remaining_time = state["remaining_time"]
    prev_actions = state["selected_activities"]
    last_turn = state["turn"]
    
    duration = activities[activity]["duration"]
    
    # find then time remainaing after processing the activity decided to perform 
    new_remaining_time = remaining_time - duration
    
    # find the next turn 
    next_turn = findNextTurn(last_turn)
    
    new_state = {
        "remaining_time" : new_remaining_time ,
        "selected_activities" : prev_actions + [activity], 
        "turn" : next_turn ,
        "players" : n
    }    
    
    return new_state



# counter to count nodes 
maxN_nodes = 0 
maxN_evaluated = 0 

def maxN(curr_state , curr_path ):
    global maxN_evaluated
    global maxN_nodes 
    maxN_nodes += 1

    # base case  : 
    if isTerminal(curr_state):
        maxN_evaluated += 1
        utility = terminalUtility(curr_state)
        return utility ,curr_path
    
    # get all teh legal activites that they can perform 
    legal_actions  = getLegalActions(curr_state)
    
    player = curr_state["turn"]
    
    best_utility = None
    best_path = None 
        
    for action in legal_actions: 
        child_state = makeMove(curr_state,action)                
        new_path = curr_path + [action]
        utility , path = maxN(child_state , new_path)
        
        if best_utility is None: 
            best_utility = utility
            best_path = path
        
        elif utility[player] > best_utility[player]: 
            best_utility = utility
            best_path = path
            
    return best_utility , best_path
        
            
def run_demo():
    test_state = {
        "remaining_time": 16,
        "selected_activities": [],
        "turn": 0,
        "players": 4
    }

    global maxN_nodes, maxN_evaluated
    maxN_nodes = 0
    maxN_evaluated = 0

    print("test_state is given as ", test_state)
    value1, path1 = maxN(test_state, [])
    print("")
    print("Max-N utility:", value1)
    print("Max-N path:", path1)
    print("Max-N nodes:", maxN_nodes)
    print("Max-N evaluated:", maxN_evaluated)


if __name__ == "__main__":
    run_demo()
