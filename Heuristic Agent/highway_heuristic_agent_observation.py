import gymnasium as gym 
import highway_env 
import numpy as np 

env = gym.make("highway-v0", render_mode = "human")
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

env.unwrapped.configure(config)
obs, info = env.reset() 

done = False 
running = True
#print("observations: " + str(obs))
print("observation shape: " + str(obs.shape))
while running and not done: 

    presence = obs[0,...]
    print("presence array: \n" + str(presence))

    front_slice = presence[4:6, 6:11] 
    left_slice = presence[1:4, 3:9] 
    right_slice = presence[6:9, 3:9] 

    print("front slice: " + str(front_slice))
    print("left slice: " + str(left_slice))
    print("right slice: " + str(right_slice))

    carInFront = np.any(front_slice) 
    carOnLeft = np.any(left_slice)
    carOnRight = np.any(right_slice)

    print("car in front: " + str(carInFront))
    print("car on left: " + str(carOnLeft))
    print("car on right: " + str(carOnRight))

    print("y coordinate: " + str(env.unwrapped.vehicle.position[1]))

    if carInFront == True and carOnLeft == True and carOnRight == True: 
        print("car in front and on left and on right - slow ")
        action = 4 #slow 
        numSlow += 1
    elif carInFront and (env.unwrapped.vehicle.position[1] >= 0 and 
                         env.unwrapped.vehicle.position[1] <= 1) and not carOnRight: 
        print("car in front and on left row - turn right")
        action = 2 # turn right
        numRight += 1
    elif carInFront and (env.unwrapped.vehicle.position[1] <= 12 and 
                         env.unwrapped.vehicle.position[1] >= 11) and not carOnLeft:
        print("car in front and on right lane - left") 
        action = 0 
        numLeft += 1
    elif carOnLeft and carOnRight and not carInFront and not np.any(presence[0:4, 4:11]): #
        print("car on left and right but not front - speed ")
        action = 3 #speed up 
        numAccelerate += 1 
    elif carInFront and carOnRight and not carOnLeft and not (env.unwrapped.vehicle.position[1] >= 0 and 
                         env.unwrapped.vehicle.position[1] <= 1):
        print("car in front and car on right - turn left")
        action = 0 #turn left 
        numLeft += 1
        action = 4 
        numSlow += 1
    elif carInFront and carOnLeft and not carOnRight and not (env.unwrapped.vehicle.position[1] <= 12 and 
                         env.unwrapped.vehicle.position[1] >= 11): 
        print("car in front and on left - turn right")
        action = 2 # turn right 
        numRight += 1
        action = 4 
        numSlow += 1
    elif carInFront: 
        print("car in front - slow")
        action = 4 # slow 
        numSlow += 1
    else: 
        if carInFront: 
            action = 4
            numSlow += 1
        
        print("idle/slow")
        action = 1
        #numSlow +=1 
        numIdle += 1
    print("action: " + str(action))
    obs, reward, done, truncated, info = env.step(action)
    env.render()

    totalDistance += reward 
    totalSpeed += env.unwrapped.vehicle.speed
    numSteps += 1

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
    print("num idle: " + str(numIdle))
    print()

env.close()
    