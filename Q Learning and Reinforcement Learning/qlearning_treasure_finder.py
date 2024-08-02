import numpy as np 
import random 
import time 
import os 

# grid_size = 10 
# obstacle_prob = 0.2 
# goal_reward = 100 
# obstacle_reward = -100 
# empty_reward = -1 

# Q-learning parameters
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.1  # Exploration rate
episodes = 1000  # Number of episodes
update_interval = 50 

actions = ["up", "down", "left", "right"]
action_mapping = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

np.random.seed(0)
env = np.zeros((10, 10)) #create 10x10 grid with 0s 
env[9,9] = 2 #treasure in corner 

#place obstacles randomly 
for i in range(10): 
    for j in range(10): 
        if (i,j) != (0,0) and (i,j) != (9,9) and np.random.rand() < 0.2: 
            #only place in obstacle if its not at (0,0), not in the treasure location 
            #and if the random number generates less than 0.2 (20% chance)
            env[i,j] = 1

Q_table = np.zeros((10,10,len(actions)))

def is_valid_position(position): 
    return 0 <= position[0] < 10 and 0 <= position[1] < 10

def get_reward(position): 
    if env[position] == 2: 
        return 100 #if we reach the treasure, reward is 100
    elif env[position] == 1: 
        return -100 #if hit obstacle, reward is -100
    else: 
        return -1 #if anything else (empty space), reward is -1 
def print_enviroment(env, state): 
    for i in range(10): 
        for j in range(10): 
            if(i,j) == state: 
                print("X", end = " ")
            elif env[i,j] == 2: 
                print("2", end = " ")
            elif env[i,j] == 1: 
                print("1", end = " ")
            else: 
                print("0", end = " ")
        print()
    print("\n")

def print_Q_table(qTable): 
    print("Q table: ")
    print("Up            Down        Left        Right")
    for i in range(10): 
        for j in range(10): 
            print(f"({i},{j}): {qTable[i, j, :]}")
    print("\n")

#training 
for episodes in range(1000): 
    state = (0,0)
    done = False 

    while not done: 
        action = actions[np.argmax(Q_table[state[0], state[1], :])]
        next_state = (state[0] + action_mapping[action][0], state[1] + action_mapping[action][1])

        if not is_valid_position(next_state): 
            next_state = state #if not a valid position, stay 
        
        reward = get_reward(next_state)

        #update the Q table 
        alpha = 0.1  # Learning rate
        gamma = 0.9  # Discount factor  

        #calculate q table + update 
        Q_table[state[0], state[1], actions.index(action)] += alpha * (reward + gamma * np.max(Q_table[next_state[0], next_state[1], :])
                                                                       - Q_table[state[0], state[1], actions.index(action)])
        state = next_state
                  
        if episodes % update_interval == 0: 
            print_enviroment(env, state)
            #print_Q_table(Q_table)
            time.sleep(0.2) # pause 

        if state == (9,9): 
            done = True
            break 

optimal_policy = np.full((10,10), "", dtype = "object")
for i in range(10): 
    for j in range(10): 
        if env[i,j] == 0 or env[i,j] == 2: 
            optimal_policy[i,j] = actions[np.argmax(Q_table[i,j,:])]

print("Enviroment: ")
print(env)
print("optimal policy: ")
print(optimal_policy)
print("Q-Table")
print("Up            Down        Left        Right")
print(Q_table)

