# lets try assingning eachc activity valuse according to A and B in a pair (x,y) 
# where x is the score by A on an actiivity and B is the score by B on the same activity. 
# The scores are based on how much they enjoy or prefer that activity.

TOTAL_TIME = 24 # ON SUNDAY I HAVE TOTAL 24 HRS

# for now we have only 2 players A and B, we can add more players later if needed.
activities = {
    "Coding": {
        "duration" : 2 , 
        "utility": (9, 4)},
    "Exam Prep": {
        "duration" : 2 ,
        "utility": (8, 6)},
    "Research": {
        "duration" : 3 ,
        "utility": (8, 5)},
    "Sleep": {
        "duration" : 7 ,
        "utility": (3, 9)},
    "Cooking": {
        "duration" : 2 ,
        "utility": (4, 6)},
    "Eat Outside": {
        "duration" : 2 ,
        "utility": (5, 8)},
    "Movie": {
        "duration" : 2 ,
        "utility": (5, 8)},
    "Games": {
        "duration" : 2 ,
        "utility": (6, 7)},
    "Reading": {
        "duration" : 2 ,
        "utility": (6, 3)},
}

state = {
    "remaining_time": TOTAL_TIME,
    "selected_activities": [],
    "turn" : "A"
}


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
    u_a  = 0 
    u_b =  0
    
    leagalActivities = state["selected_activities"]
    for activity in leagalActivities: 
        a , b = activities[activity]["utility"]
        
        u_a += a
        u_b += b
        
    
    return u_a - u_b 


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

    # right now i am only conisdering 2 players 
    # Since we currently have only two players:
    # A -> B
    # B -> A
    def findNextTurn(X): 
        return "B" if X == "A" else "A" 
    
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
        "turn" : next_turn
    }    
    
    return new_state



# counter to count nodes 
minimax_nodes = 0 
minimax_evaluated = 0 

def minimax(curr_state , curr_path ):
    global minimax_evaluated
    global minimax_nodes 
    minimax_nodes += 1

    # base case  : 
    if isTerminal(curr_state):
        minimax_evaluated += 1
        value = terminalUtility(curr_state)
        return value ,curr_path
    # get all teh legal activites that they can perform 
    legal_actions  = getLegalActions(curr_state)

    #if player is A then we need to MAX else MIN  
    # MAX 
    if curr_state["turn"] == "A":
        best_value = float("-inf")
        best_path = None
        
        for action in legal_actions: 
            child_state = makeMove(curr_state , action)
            new_path = curr_path + [action]
            value , path = minimax(child_state , new_path)
            # print("Optimal Value" , value)
            if value > best_value: 
                best_value = value
                best_path = path
            
        return best_value , best_path

    # min
    else : 
        best_value = float("inf")
        best_path = None
                
        for action in legal_actions: 
            child_state = makeMove(curr_state , action)
            new_path = curr_path + [action]
            value , path = minimax(child_state , new_path)
            
            if value < best_value :
                best_value = value
                best_path = path
                
        return best_value , best_path
        
        
        
        
        
        
# Trying to code alpha - beta pruning 
# pruning -- > if we know this branch exploring is not good then we stop 


# counter to count alpha_beta nodes 
alpha_beta_nodes = 0 
# to show the pruned path
alpha_beta_pruned = 0

alpha_beta_evaluated = 0  
def alphaBetaPruning(curr_state , curr_path , alpha , beta) : 
    
    global alpha_beta_nodes , alpha_beta_pruned , alpha_beta_evaluated
    alpha_beta_nodes += 1    
    if isTerminal(curr_state):
        alpha_beta_evaluated += 1
        return terminalUtility(curr_state) , curr_path
    
    legal_actions = getLegalActions(curr_state)
    
    # MAX
    if curr_state["turn"] == "A": 
        
        best_value = float("-inf") 
        best_path = None 
        
        for action in legal_actions: 
            child_state = makeMove(curr_state, action)
            new_path = curr_path + [action]
            value , path = alphaBetaPruning(child_state , new_path ,alpha , beta)
            
            if value > best_value : 
                best_value = value
                best_path = path
            
            alpha = max(alpha , best_value)
            
            if alpha >= beta : 
                alpha_beta_pruned += 1 
                # print(
                #     "PRUNED:",
                #     curr_path,
                #     "| alpha =", alpha,
                #     "| beta =", beta
                #      )

                break
                        
                
        return best_value , best_path
    
    # MIN
    else :
        best_value = float("inf") 
        best_path = None 
        
        for action in legal_actions: 
            child_state = makeMove(curr_state, action)
            new_path = curr_path + [action]
            value , path = alphaBetaPruning(child_state , new_path ,alpha , beta)
            
            if value < best_value : 
                best_value = value
                best_path = path
            
            beta = min(beta , best_value)
            if alpha >= beta : 
                alpha_beta_pruned += 1 
                # print(
                #         "PRUNED:",
                #         curr_path,
                #         "| alpha =", alpha,
                #         "| beta =", beta
                #     )

                break
                
        return best_value , best_path
        
    
def run_demo():
    test_state = {
        "remaining_time": 24,
        "selected_activities": [],
        "turn": "A"
    }

    global minimax_nodes, minimax_evaluated, alpha_beta_nodes, alpha_beta_pruned, alpha_beta_evaluated
    minimax_nodes = 0
    minimax_evaluated = 0
    alpha_beta_nodes = 0
    alpha_beta_pruned = 0
    alpha_beta_evaluated = 0

    print("test_state is given as ", test_state)
    value1, path1 = minimax(test_state, [])
    print("")
    print("Minimax utility:", value1)
    print("Minimax path:", path1)
    print("Minimax nodes:", minimax_nodes)
    print("Minimax evaluated:", minimax_evaluated)

    print("")

    value2, path2 = alphaBetaPruning(
        test_state,
        [],
        float("-inf"),
        float("inf")
    )
    print("")
    print("Alpha-Beta utility:", value2)
    print("Alpha-Beta path:", path2)
    print("Alpha-Beta nodes:", alpha_beta_nodes)
    print("Alpha-Beta evaluated:", alpha_beta_evaluated)
    print("Alpha-Beta pruning events:", alpha_beta_pruned)


if __name__ == "__main__":
    run_demo()