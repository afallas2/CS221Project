import gym
import time
env = gym.make('LunarLander-v2')
env.reset()
for _ in range(1000):
    env.render()
    observation, reward, done, info = env.step(env.action_space.sample())
    posX = observation[0]
    print("posX: {}".format(posX))
    posY = observation[1]
    print("posY: {}".format(posY))
    velX = observation[2]
    print("X Velocity: {}".format(velX))
    velY = observation[3]
    print("Y Velocity: {}".format(velY))
    time.sleep(.1)
