import gymnasium as gym 
import highway_env
import pygame 
from gymnasium.utils.play import play
#from pygame import event 
#from pygame.locals import *



# stdscr = curses.initscr()
# curses.noecho()
# curses.cbreak()
# stdscr.keypad(True)
config = {
    "lanes_count" : 4,
    "vehicles_count" : 50,
    "duration" : 100, 
    "initial_spacing":2,
    "simulation_frequency":15, 
    "policy_frequency": 0.25,
    "render_agent": True, 
}

pygame.init()

screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption("Highway Environment Control")

running = True 



env = gym.make("highway-v0", render_mode = "rgb_array") #
#env.get_wrapper_attr('vehicle')
#env.unwrapped.vehicle
#env.unwrapped.configure({"manual_control" : True})
env.configure ({
     "action":{
         "type":"DiscreteMetaAction",
         "action_config" : {
             "type": "DiscreteMetaAction",
         }
     },
     "manual_control" : True
 })
env.reset()




totalCrashes = 0 
totalDistance = 0 
totalSpeed = 0
collisionFreeSpeed = []
actions = []
numSteps = 0 
numLeft = 0 
numRight = 0 
numAccelerate = 0 
numSlow = 0 

action = 1 


totalDistance = 0 
totalSpeed = 0 
action = 0

def manual_callback(obs_t, obs_tp1, action, rew, terminated, truncated, info): 
    print("obs_t: " + str(obs_t) + " obs_tp1: " + str(obs_tp1) + " action: " + str(action)+ " rew: " + str(rew) + 
          " terminated: "+ str(terminated) + " truncated: " + str(truncated) + " info: " + str(info) )
    
    global totalDistance
    global totalSpeed
    global totalCrashes
    global collisionFreeSpeed
    global numSteps 

    global numLeft
    global numRight
    global numAccelerate
    global numSlow
 

    totalDistance += rew 
    totalSpeed += env.unwrapped.vehicle.speed
    numSteps += 1
    #print("total speed: " + str(totalSpeed))

    if env.unwrapped.vehicle.crashed:
        totalCrashes += 1
    else:
        collisionFreeSpeed.append(env.unwrapped.vehicle.speed)

    avgSpeed = totalSpeed/numSteps 
    avgCollisionFreeSpeed = (sum(collisionFreeSpeed)/len(collisionFreeSpeed)) if collisionFreeSpeed else 0 
    #collisionsPer1000 = totalCrashes / (totalDistance/1000)

    print("reward: " + str(rew))
    print("average speed: " + str(avgSpeed))
    print("average collision free speed: " + str(avgCollisionFreeSpeed))
    print("total collision: " + str(totalCrashes))

    print("action: " + str(type(action)) + " Value = " + str(action))
    if action == 0: 
        numLeft += 1
    if action == 2:
        numRight += 1
    if action == 3:
        numAccelerate += 1
    if action == 4:
        numSlow += 1
    print("numLeft: " + str(numLeft))
    print("numRight: " + str(numRight))
    print("numAccelerate: " + str(numAccelerate))
    print("numSlow: " + str(numSlow))

    if (terminated):
        env.close()
        pygame.quit()  
        


mapping = {(pygame.K_UP,) : 0, (pygame.K_DOWN,) : 1, (pygame.K_RIGHT,) : 2, (pygame.K_LEFT,) : 3, ("s"):4}
#mapping = {("x") : 100, ("x") : 200, ("d") : 300, ("a") : 400}
#mapping = {("a") : 0, ("s") : 1, ("d") : 2, ("f") : 3, ("g"):4}


#play(env, callback = manual_callback, keys_to_action = highway_env.envs.common.action.DiscreteMetaAction.ACTIONS_ALL) 
play(env, callback = manual_callback, keys_to_action = mapping) 

# while running: 
    
#     #print("inside while")
#     print("pygame event get: " + str(pygame.event.get()))
#     # for event in pygame.event.get():
#     #     print("pygame event get: " + str(event.type))

#     #     if event.type == pygame.QUIT:
#     #          pygame.quit()
#     #          running = False 
#     #     if event.type == pygame.KEYDOWN:
#     #         print("coming in")
#     #         if event.key == pygame.K_LEFT: 
#     #             print("turnin left")
#     #             action = 0 #left 
#     #             numLeft += 1
#     #         if event.key == pygame.K_RIGHT:
#     #             print("turnin right")
#     #             action = 2 #right 
#     #             numRight += 1
#     #         if event.key == pygame.K_UP:
#     #             print("accelerating")
#     #             action = 3 #go faster 
#     #             numAccelerate += 1
#     #         if event.key == pygame.K_DOWN: 
#     #             print("descelerating")
#     #             action = 4 #slow/stop 
#     #             numSlow += 1
#     #     elif event.type == pygame.KEYUP:
#     #         action = 1 #do nothing
#     print("action: " + str(action)):1
#     obs, reward, done, truncated, info = env.step(action)
#     frame = env.render()

#     totalDistance += reward 
#     totalSpeed += env.unwrapped.vehicle.speed
#     #actions.append(action)
#     numSteps += 1 

#     if env.vehicle.crashed:
#         totalCrashes += 1
#     else:
#         collisionFreeSpeed.append(env.unwrapped.vehicle.speed)


#     if done or truncated:   
#         obs, info = env.reset()
    
#     avgSpeed = totalSpeed/numSteps 
#     avgCollisionFreeSpeed = (sum(collisionFreeSpeed)/len(collisionFreeSpeed)) if collisionFreeSpeed else 0 
#     collisionsPer1000 = totalCrashes / (totalDistance/1000)

#     #print("actions array: " + str(actions))
#     print("average speed: " + str(avgSpeed))
#     print("average collision free speed: " + str(avgCollisionFreeSpeed))
#     print("total collision: " + str(totalCrashes))
#     #print("totalDistance: " + str(totalDistance))
#     #print("collisions per 1000 meters: " + str(collisionsPer1000)) 
#     print("actions: ")
#     print("number of times turned left: " + str(numLeft))
#     print("number of times turned right: " + str(numRight))
#     print("number of times accelerated: " + str(numAccelerate))
#     print("number of times slowed down: " + str(numSlow)) 

 

    # with keyboard.Events() as events:
    #     for event in events:
    #         if event.key == keyboard.Key.up:
    #             numLeft += 1
    #         elif event.key == keyboard.Key.down: 
    #             numRight += 1
    #         elif event.key == keyboard.Key.right: 
    #             numAccelerate += 1
    #         elif event.key == keyboard.Key.left: 
    #             numSlow += 1
    # if keyboard.on_press("up"):
    #     numLeft += 1
    # if keyboard.on_press("down"):
    #     numRight += 1 
    # if keyboard.on_press("right"): 
    #     numAccelerate += 1
    # if keyboard.on_press("left"): 
    #     numSlow += 1
    # print("inside while")

            
