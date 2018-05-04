import gym
import time
env = gym.make('LunarLander-v2')
env.reset()
for _ in range(1000):
    env.render()
    action = input("what movement?")
    action = int(action)
    observation, reward, done, info = env.step(action)
    posX = observation[0]
    print("posX: {}".format(posX))
    posY = observation[1]
    velX = observation[2]
    velY = observation[3]

    verticalAngle = observation[4]
    angularVelocity = observation[5]
    time.sleep(.1)
    if (done):
        break    
