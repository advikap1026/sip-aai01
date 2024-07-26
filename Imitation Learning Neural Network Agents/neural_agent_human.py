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
    "duration": 100,
    "initial_spacing": 2,
    "simulation_frequency": 15,
    "policy_frequency": 0.25,
    "render_agent": True,
}

data = np.load("human_data.npz")
print("data: " + str(data))
observations = data['observations']
print("observations: "+ str(observations))
actions = data['actions']
print("actions: " + str(actions))

observations_tensor = torch.tensor(observations, dtype = torch.float32)
actions_tensor = torch.tensor(actions, dtype = torch.long)
dataset = TensorDataset(observations_tensor, actions_tensor)

train_size = int(0.7*len(dataset))
val_size = int(0.15*len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size = 32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size = 32, shuffle = False)
test_loader = DataLoader(test_dataset, batch_size = 32, shuffle = False)

class ImitationLearningAgent(nn.Module): 
    def __init__(self, input_size, output_size): 
        super(ImitationLearningAgent, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128), 
            nn.ReLU(), 
            nn.Linear(128, 128), 
            nn.ReLU(), 
            nn.Linear(128, output_size)
        )
    def forward(self, x):
        return self.network(x)

input_size = observations.shape[1]
output_size = 5 #number of discrete actions 
agent = ImitationLearningAgent(input_size, output_size)

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

    print("Epoch: " + str((epoch + 1)/num_epochs))
    print("Train Loss: " + str(train_loss/len(train_loader)))
    print("Validation Loss: " + str(val_loss/len(val_loader)))

def evaluate_agent(env, agent, num_episodes = 10): 
    total_crashes = 0 
    total_distance = 0 
    total_speed = 0 
    action_counts = [0,0,0,0,0]
    num_steps = 0 

    for i in range(num_episodes): 
        obs, info = env.reset()
        done = False
        
        while not done: 
            obs_tensor = torch.tensor(obs.flatten(), dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                action_probs = agent(obs_tensor)
                print("action probs: " + str(action_probs))
                action = torch.argmax(action_probs).item()
                print("action: " + str(action))
                #action_distribution[action] += 1
                
            obs, reward, done, truncated, info = env.step(action)
            num_steps += 1
            total_distance += reward
            total_speed += env.unwrapped.vehicle.speed
            action_counts[action] += 1

            if env.unwrapped.vehicle.crashed: 
                total_crashes += 1

    avg_speed = total_speed / num_steps
    collisions_per_1000 = total_crashes / (total_distance / 1000)
   # action_distribution = {i: count / num_steps for i, count in enumerate(action_counts)}

    return avg_speed, collisions_per_1000, action_counts, total_crashes

train_model(agent, train_loader, val_loader)
torch.save(agent.state_dict(), "imitation_agent.pth")

agent = ImitationLearningAgent(input_size, output_size)
agent.load_state_dict(torch.load("imitation_agent.pth"))
agent.eval()
env = gym.make("highway-v0", config=config, render_mode="rgb_array")
avg_speed, collisions_per_1000, action_counts, total_crashes = evaluate_agent(env, agent)

print("Average Speed: " + str(avg_speed))
print("Collissions per 1000m: " + str(collisions_per_1000))
print("total collissions: " + str(total_crashes))
print("Actions: ") 
print("Num Left: " + str(action_counts[0]))
print("Num Idle: " + str(action_counts[1]))
print("Num right: " + str(action_counts[2]))
print("Num accelerate: " + str(action_counts[3]))
print("num slow: " + str(action_counts[4]))
