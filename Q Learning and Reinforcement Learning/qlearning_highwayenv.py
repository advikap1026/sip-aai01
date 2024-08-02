import numpy as np
import gymnasium as gym
import highway_env

# Define the environment configuration
config = {
    "lanes_count": 4,
    "vehicles_count": 50,
    "duration": 100,  # [s]
    "initial_spacing": 2,
    "simulation_frequency": 15,  # [Hz]
    "policy_frequency": 0.25,  # [Hz]
    "render_agent": True,
    "observation": {
        "type": "Kinematics",
        "features": ["presence", "x", "y", "vx"],  # Using presence, x, y, and x_velocity (vx)
        "vehicles_count": 1,  # Considering only the ego vehicle
        "absolute": False
    }
}

# Initialize the environment
env = gym.make("highway-v0", config=config)

# Discretization parameters
num_bins = 3  # Reduced number of bins
observation_shape = (4,)  # Four features: presence, x, y, x_velocity
observation_bins = [
    np.linspace(0, 1, num_bins),  # Presence is binary, so 0 to 1
    np.linspace(-100, 100, num_bins),  # x-position
    np.linspace(-100, 100, num_bins),  # y-position (considering lane indices)
    np.linspace(-20, 20, num_bins)   # x_velocity
]
action_space_size = env.action_space.n

# Q-table initialization
Q = np.random.uniform(low=0, high=1, size=(num_bins,) * observation_shape[0] + (action_space_size,)) 

# Function to discretize the observation
def discretize_observation(obs): 
    state = tuple(np.digitize(obs[i], observation_bins[i]) - 1 for i in range(len(obs)))
    return state

# Function to choose an action using an ε-greedy policy
def choose_action(state, epsilon):
    if np.random.random() < epsilon:
        return np.random.choice(action_space_size)
    else:
        return np.argmax(Q[state])

# Hyperparameters
alpha = 0.1  # Learning rate
gamma = 0.99  # Discount factor
epsilon = 1.0  # Exploration rate
epsilon_decay = 0.995
min_epsilon = 0.01
num_episodes = 1000  # Number of training episodes

# Training loop
for episode in range(1000):
    obs, info = env.reset()
    obs = obs[0]  # Get the ego vehicle's observation
    state = discretize_observation(obs)
    done = False
    total_reward = 0
    
    while not done:
        action = choose_action(state, epsilon)
        obs, reward, done, truncated, info = env.step(action)
        obs = obs[0]  # Get the ego vehicle's observation
        next_state = discretize_observation(obs)
        
        # Q-learning update
        best_next_action = np.argmax(Q[next_state])
        Q[state][action] += alpha * (reward + gamma * Q[next_state][best_next_action] - Q[state][action])
        
        state = next_state
        total_reward += reward
    
    epsilon = max(min_epsilon, epsilon * epsilon_decay)
    print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")

# Evaluation
def evaluate_q_learning_agent(env, Q, num_episodes=10):
    total_rewards = []
    action_distribution = np.zeros(env.action_space.n)
    total_distance = 0 
    total_speed = 0
    total_crashes = 0 
    num_steps = 0 
    collision_free_speeds = []


    for _ in range(num_episodes):
        obs, info = env.reset()
        obs = obs[0]  # Get the ego vehicle's observation
        state = discretize_observation(obs)
        done = False
        total_reward = 0
        episode_steps = 0 
        
        
        while not done:
            action = np.argmax(Q[state])
            print("action: " + str(action))
            action_distribution[action] += 1

            obs, reward, done, truncated, info = env.step(action)
            obs = obs[0]  # Get the ego vehicle's observation

            state = discretize_observation(obs)
            total_reward += reward

            total_distance += reward 
            total_speed += env.unwrapped.vehicle.speed 
            num_steps += 1

            if env.unwrapped.vehicle.crashed:
                total_crashes += 1
            else:
                collision_free_speeds.append(env.unwrapped.vehicle.speed)
            
            if done or truncated: 
                obs, info = env.reset()

        
        total_rewards.append(total_reward)
    
    avg_reward = np.mean(total_rewards)
    avg_reward = np.mean(total_rewards)
    avg_speed = total_speed / num_steps if num_steps > 0 else 0
    avg_collision_free_speed = np.mean(collision_free_speeds) if collision_free_speeds else 0
    collisions_per_1000 = (total_crashes / (total_distance / 1000)) if total_distance > 0 else 0

    print(f"Average Reward over {num_episodes} episodes: {avg_reward}")
    print("Actions: ") 
    print("Num Left: " + str(action_distribution[0]))
    print("Num Idle: " + str(action_distribution[1]))
    print("Num right: " + str(action_distribution[2]))
    print("Num accelerate: " + str(action_distribution[3]))
    print("num slow: " + str(action_distribution[4]))
    print(f"Average Speed: {avg_speed}")
    print(f"Average Collision-Free Speed: {avg_collision_free_speed}")
    print(f"Total Collisions: {total_crashes}")
    print(f"Collisions per 1000 meters: {collisions_per_1000}")

    return avg_reward, action_distribution, avg_speed, avg_collision_free_speed, total_crashes, collisions_per_1000
    #return avg_reward

# Evaluate the trained agent
avg_reward, action_counts, avg_speed, avg_collision_free_speed, total_crashes, collisions_per_1000 = evaluate_q_learning_agent(env, Q)
print(f"Average Reward over {num_episodes} episodes: {avg_reward}")

env.close()