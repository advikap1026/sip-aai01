import gymnasium as gym 
import highway_env
import pygame 
from pynput import keyboard 
#from pygame import event 
#rom gymnasium.utils.play import play
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

running = True 


env = gym.make("highway-v0", render_mode = "human") 

pygame.init()

screen = pygame.display.set_mode((600, 150))
pygame.display.set_caption("Highway Environment Control")
clock = pygame.time.Clock()
#env.get_wrapper_attr('vehicle')
#env.unwrapped.vehicle
env.unwrapped.configure({"manual_control" : True})
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

while running: 
    # screen.fill((0, 0, 0))
    print("inside while")
    event = pygame.event.get()
    print(str(event))
    keys = pygame.key.get_pressed()
    print("key: " + str(keys[pygame.K_UP]))
    #print("key pressed: " + str(pygame.key.get_pressed[pygame.K_UP]))
    print("pygame event get: " + str(pygame.event.get())) 
    if keys[pygame.K_UP] == True: 
        numLeft += 1
    elif keys[pygame.K_DOWN] == True:
        numRight += 1
    elif keys[pygame.K_RIGHT] == True:
        numAccelerate += 1
    elif keys[pygame.K_LEFT] == True: 
        numSlow += 1
    
    # for event in pygame.event.get():
    #     print("pygame event get: " + str(event.type))  

    #     # if event.type == pygame.QUIT:
    #     #      pygame.quit()
    #     #      running = False 
    #     if event.type == pygame.KEYDOWN:
    #         print("coming in")
    #         if event.key == pygame.K_LEFT: 
    #             print("turning left")
    #             action = 0 #left 
    #             numLeft += 1
    #         if event.key == pygame.K_RIGHT:
    #             print("turning right")
    #             action = 2 #right 
    #             numRight += 1
    #         if event.key == pygame.K_UP:
    #             print("accelerating")
    #             action = 3 #go faster 
    #             numAccelerate += 1
    #         if event.key == pygame.K_DOWN: 
    #             print("descelerating")
    #             action = 4 #slow/stop 
    #             numSlow += 1
    #     elif event.type == pygame.KEYUP:
    #         action = 1 #do nothing
    print("action: " + str(action))
    obs, reward, done, truncated, info = env.step(action)
    frame = env.render()

    # pygame.surfarray.blit_array(screen, frame.swapaxes(0, 1))
    # pygame.display.flip()
    # clock.tick(15)

    totalDistance += reward 
    totalSpeed += env.unwrapped.vehicle.speed
    #actions.append(action)
    numSteps += 1 

    if env.vehicle.crashed:
        totalCrashes += 1
    else:
        collisionFreeSpeed.append(env.unwrapped.vehicle.speed)


    if done or truncated:   
        obs, info = env.reset()
    
    avgSpeed = totalSpeed/numSteps 
    avgCollisionFreeSpeed = (sum(collisionFreeSpeed)/len(collisionFreeSpeed)) if collisionFreeSpeed else 0 
    collisionsPer1000 = totalCrashes / (totalDistance/1000)

    #print("actions array: " + str(actions))
    print("average speed: " + str(avgSpeed))
    print("average collision free speed: " + str(avgCollisionFreeSpeed))
    print("total collision: " + str(totalCrashes))
    #print("totalDistance: " + str(totalDistance))
    #print("collisions per 1000 meters: " + str(collisionsPer1000)) 
    print("actions: ")
    print("number of times turned left: " + str(numLeft))
    print("number of times turned right: " + str(numRight))
    print("number of times accelerated: " + str(numAccelerate))
    print("number of times slowed down: " + str(numSlow)) 

env.close()
pygame.quit()   

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

            
