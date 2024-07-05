import gymnasium as gym
import highway_env
import numpy as np 

# Create the environment with a specified render mode
env = gym.make("highway-v0", render_mode="human")
config = {
    "lanes_count" : 4,
    "vehicles_count" : 50,
    "duration" : 100, 
    "initial_spacing":2,
    "simulation_frequency":15, 
    "policy_frequency": 0.25,
    "render_agent": True, 

}
env.configure(config)
# Initialize the environment
env.reset()

totalCrashes = 0 
totalDistance = 0 
numSteps = 0 
totalSpeed = 0
collisionFreeSpeed = []


numLeft = 0 
numRight = 0 
numAccelerate = 0
numSlow = 0 
numIdle = 0 

done = False
running = True 
while running or not done:
    
    action = env.action_space.sample()
    print("action: " + str(action)) 
    obs, reward, done, truncated, info = env.step(action)

    if action == 0:
        numLeft += 1
    if action == 1:
        numIdle += 1
    if action == 2:
        numRight += 1
    if action == 3: 
        numAccelerate += 1
    if action == 4: 
        numSlow += 1

    
    
    # Render the environment
    env.render()

    totalDistance += reward 
    totalSpeed += env.unwrapped.vehicle.speed
    numSteps += 1
    #actions.append(action)
    #numSteps += 1

    if env.unwrapped.vehicle.crashed:
        totalCrashes += 1
    else:
        collisionFreeSpeed.append(env.unwrapped.vehicle.speed)

    if done or truncated:   
        obs, info = env.reset()
    
    avgSpeed = totalSpeed/numSteps 
    avgCollisionFreeSpeed = (sum(collisionFreeSpeed)/len(collisionFreeSpeed)) if collisionFreeSpeed else 0 
    collisionsPer1000 = totalCrashes / (totalDistance/1000)

       
    print("average speed: " + str(avgSpeed))
    print("average collision free speed: " + str(avgCollisionFreeSpeed))
    print("total collision: " + str(totalCrashes))

    print()
    print("actions: ")
    print("num left: " + str(numLeft))
    print("num right: " + str(numRight))
    print("num accelerate: " + str(numAccelerate))
    print("num slow: " + str(numSlow))
    print()


# Close the environment
env.close()