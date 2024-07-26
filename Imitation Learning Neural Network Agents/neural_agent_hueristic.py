import gymnasium as gym 
import highway_env 
import torch 
import torch.nn as nn 
import torch.optim as optim 
from torch.utils.data import DataLoader, TensorDataset, random_split 
import numpy as np 

config = {
    "lanes_count": 4,
    "vehicles_count": 50,
    "duration": 100,  # [s]
    "initial_spacing": 2,
    "simulation_frequency": 15,  # [Hz]
    "policy_frequency": 0.25,  # [Hz]
    "render_agent": True,
    "observation": {
        "type": "OccupancyGrid",
        "vehicles_count": 50,
        "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
        "features_range": {
            "x": [-100, 100],
            "y": [-100, 100],
            "vx": [-20, 20],
            "vy": [-20, 20]
        },
        "grid_size": [[-27.5, 27.5], [-27.5, 27.5]],
        "grid_step": [5, 5],
        "absolute": False
    }
}

env = gym.make("highway-fast-v0", config=config, render_mode='rgb_array')

def heuristic_action(obs):
    presence = obs[0, ...]
    front_slice = presence[4:6, 6:11] 
    left_slice = presence[1:4, 3:9] 
    right_slice = presence[6:9, 3:9]

    carInFront = np.any(front_slice)
    carOnLeft = np.any(left_slice)
    carOnRight = np.any(right_slice)

    if carInFront and carOnLeft and carOnRight:
        action = 4  # slow
    elif carInFront and (env.unwrapped.vehicle.position[1] >= 0 and env.unwrapped.vehicle.position[1] <= 1) and not carOnRight:
        action = 2  # turn right
    elif carInFront and (env.unwrapped.vehicle.position[1] <= 12 and env.unwrapped.vehicle.position[1] >= 11) and not carOnLeft:
        action = 0  # turn left
    elif carOnLeft and carOnRight and not carInFront and not np.any(presence[0:4, 4:11]):
        action = 3  # speed up
    elif carInFront and carOnRight and not carOnLeft and not (env.unwrapped.vehicle.position[1] >= 0 and env.unwrapped.vehicle.position[1] <= 1):
        action = 0  # turn left
    elif carInFront and carOnLeft and not carOnRight and not (env.unwrapped.vehicle.position[1] <= 12 and env.unwrapped.vehicle.position[1] >= 11):
        action = 2  # turn right
    elif carInFront:
        action = 4  # slow
    else:
        action = 1  # idle
    return action

def collect_data(env, num_episodes = 10):
    observations=[]
    actions = []
    for episodes in range(num_episodes):
        obs, info = env.reset()
        done = False 
        while not done: 
            action = heuristic_action(obs)
        
            observations.append(obs.flatten()) 
            actions.append(action) 
            obs, reward, done, truncated, info = env.step(action)
            if done or truncated:
                obs, info = env.reset()
        #return torch.tensor(observations, dtype=torch.float32), torch.tensor(actions, dtype=torch.long)
        return np.array(observations), np.array(actions)
    

class ImitationLearningAgent(nn.Module):

    def __init__(self, input_size, output_size):
        super(ImitationLearningAgent, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

def train_model(agent, train_loader, val_loader, num_epochs = 50, learning_rate = 0.001): 
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(agent.parameters(), lr=learning_rate)

    for epoch in range(num_epochs): 
        agent.train()
        train_loss = 0 
        for observations, actions in train_loader: 
            optimizer.zero_grad()
            outputs = agent(observations)
            loss = criterion(outputs, actions)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        val_loss = 0 
        agent.eval() 
        with torch.no_grad():
            for observations, actions in val_loader:
                outputs = agent(observations)
                loss = criterion(outputs, actions)
                val_loss += loss.item()
        print("Epoch: " + str((epoch + 1)/(num_epochs)))
        print("Train Loss: " + str(train_loss/len(train_loader)))
        print("Validation loss: " + str(val_loss/len(val_loader)))

# def evaluate_model(agent, test_loader):

#     agent.eval()
#     all_predicted_actions = []
#     action_counts = [0,0,0,0,0] #left, idle, right, speed, slow 

#     total_distance = 0 
#     total_speed = 0 
#     total_crashes = 0 
#     num_steps = 0 
#     collision_free_speed = []


#     print("in torch")
#     with torch.no_grad(): 
#         for observations, _ in test_loader: 
#             outputs = agent(observations)
#             _, predicted_actions = torch.max(outputs, 1)
#             all_predicted_actions.extend(predicted_actions.cpu().numpy())
#             for action in predicted_actions.cpu().numpy():
#                 #print("action: " + str(action))
#                 action_counts[action] += 1
#                 obs, reward, done, truncated, info = env.step(action)
#                 num_steps += 1
#                 total_distance += reward
#                 total_speed += env.unwrapped.vehicle.speed

#                 if env.unwrapped.vehicle.crashed:
#                     total_crashes += 1
#                 else:
#                     collision_free_speed.append(env.unwrapped.vehicle.speed)

#                 if done or truncated:
#                     env.reset()
#     avg_speed = total_speed / num_steps if num_steps > 0 else 0
#     avg_collision_free_speed = (sum(collision_free_speed) / len(collision_free_speed)) if collision_free_speed else 0
#     collisions_per_1000 = total_crashes / (total_distance / 1000) if total_distance > 0 else 0

#     print("average speed: " + str(avg_speed))
#     print("average collission free speed: " + str(avg_collision_free_speed))
#     print("number of collissions per 1000m: " + str(collisions_per_1000))

#     return all_predicted_actions, action_counts

def evaluate_model(agent, test_loader, env):
    agent.eval()
    all_predicted_actions = []
    all_actual_actions = []
    action_counts = [0, 0, 0, 0, 0]  # left, idle, right, speed, slow
    correct_predictions = 0
    total_predictions = 0

    total_distance = 0 
    total_speed = 0 
    total_crashes = 0 
    num_steps = 0 
    collision_free_speed = []

    with torch.no_grad():
        for observations, actions in test_loader:
            outputs = agent(observations)
            _, predicted_actions = torch.max(outputs, 1)

            all_predicted_actions.extend(predicted_actions.cpu().numpy())
            all_actual_actions.extend(actions.cpu().numpy())

            for action in predicted_actions.cpu().numpy():
                action_counts[action] += 1

            correct_predictions += (predicted_actions == actions).sum().item()
            total_predictions += actions.size(0)

            # Evaluate in the environment
            for obs, action in zip(observations, predicted_actions):
                env.reset()
                obs, reward, done, truncated, info = env.step(action.item())
                num_steps += 1
                total_distance += reward
                total_speed += env.unwrapped.vehicle.speed

                if env.unwrapped.vehicle.crashed:
                    total_crashes += 1
                else:
                    collision_free_speed.append(env.unwrapped.vehicle.speed)

                if done or truncated:
                    env.reset()

    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    avg_speed = total_speed / num_steps if num_steps > 0 else 0
    avg_collision_free_speed = sum(collision_free_speed) / len(collision_free_speed) if collision_free_speed else 0
    collisions_per_1000 = total_crashes / (total_distance / 1000) if total_distance > 0 else 0

    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Average Speed: {avg_speed}")
    print(f"Average Collision-Free Speed: {avg_collision_free_speed}")
    print(f"Collisions per 1000m: {collisions_per_1000}")

    return all_predicted_actions, action_counts, accuracy

    # agent.eval()
    # all_predicted_actions = []
    # all_actual_actions = []
    # action_counts = [0, 0, 0, 0, 0]  # left, idle, right, speed, slow

    # correct_predictions = 0
    # total_predictions = 0

    # with torch.no_grad():
    #     for observations, actions in test_loader:
    #         outputs = agent(observations)
    #         _, predicted_actions = torch.max(outputs, 1)

    #         all_predicted_actions.extend(predicted_actions.cpu().numpy())
    #         all_actual_actions.extend(actions.cpu().numpy())

    #         for action in predicted_actions.cpu().numpy():
    #             action_counts[action] += 1

    #         correct_predictions += (predicted_actions == actions).sum().item()
    #         total_predictions += actions.size(0)

    # accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    # #print("accuracy: " + str(accuracy)) 

    # return all_predicted_actions, action_counts, accuracy


train_observations_heuristic, train_actions_heuristic = collect_data(env, num_episodes=50)
test_observations_heuristic, test_actions_heuristic = collect_data(env, num_episodes=10) 

print("observations: " + str(train_observations_heuristic))
print("actions: " + str(train_actions_heuristic))

observations_tensor = torch.tensor(train_observations_heuristic, dtype=torch.float32)
print("observations tensor dataset: " + str(observations_tensor))
actions_tensor = torch.tensor(train_actions_heuristic, dtype=torch.long) 
print("actions tensor dataset: " + str(actions_tensor))

# Create the dataset 
dataset = TensorDataset(observations_tensor, actions_tensor)
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size]) 

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

input_size = train_observations_heuristic.shape[1]
output_size = env.action_space.n 

agent_heuristic = ImitationLearningAgent(input_size, output_size)
predicted_actions_heuristic, action_counts_heuristic = evaluate_model(agent_heuristic, test_loader)

train_model(agent_heuristic, train_loader, val_loader)
predicted_actions_heuristic, action_counts_heuristic = evaluate_model(agent_heuristic, test_loader)

print("Heuristic Action Counts: ")
print("Num Left: " + str(action_counts_heuristic[0]))
print("Num Idle: " + str(action_counts_heuristic[1]))
print("Num right: " + str(action_counts_heuristic[2]))
print("Num accelerate: " + str(action_counts_heuristic[3]))
print("num slow: " + str(action_counts_heuristic[4]))

#print("accuracy: " + str(accuracy))

env.close()