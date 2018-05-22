import gym
import time
import numpy as np
from collections import defaultdict
import math, random

def heuristic(env, s):
    # Heuristic for:
    # 1. Testing. 
    # 2. Demonstration rollout.
    angle_targ = s[0]*0.5 + s[2]*1.0         # angle should point towards center (s[0] is horizontal coordinate, s[2] hor speed)
    if angle_targ >  0.4: angle_targ =  0.4  # more than 0.4 radians (22 degrees) is bad
    if angle_targ < -0.4: angle_targ = -0.4
    hover_targ = 0.55*np.abs(s[0])           # target y should be proporional to horizontal offset

    # PID controller: s[4] angle, s[5] angularSpeed
    angle_todo = (angle_targ - s[4])*0.5 - (s[5])*1.0
    #print("angle_targ=%0.2f, angle_todo=%0.2f" % (angle_targ, angle_todo))

    # PID controller: s[1] vertical coordinate s[3] vertical speed
    hover_todo = (hover_targ - s[1])*0.5 - (s[3])*0.5
    #print("hover_targ=%0.2f, hover_todo=%0.2f" % (hover_targ, hover_todo))

    if s[6] or s[7]: # legs have contact
        angle_todo = 0
        hover_todo = -(s[3])*0.5  # override to reduce fall speed, that's all we need after contact

    a = 0
    if hover_todo > np.abs(angle_todo) and hover_todo > 0.05: a = 2
    elif angle_todo < -0.05: a = 3
    elif angle_todo > +0.05: a = 1
    return a


class QLearningAlgorithm():
    def __init__(self, actions, discount, featureExtractor, explorationProb):
        self.actions = actions
        self.discount = discount
        self.featureExtractor = featureExtractor
        self.explorationProb = explorationProb
        self.weights = defaultdict(float)
        self.numIters = 0
        self.discount = discount
        self.stepSize = 1


    # Return the Q function associated with the weights and features
    def getQ(self, state, action):
        score = 0.0
        for f, v in self.featureExtractor(state, action):
            if math.isnan(self.weights[f]):
                score += float(v)
            else:
                score += float(self.weights[f] * v)
        if ((score) > (2 ** 31 - 1)):
            
            return (2 ** 31 - 1)
        if score < (-1*(2 ** 31 - 1)):
            return -(2 ** 31 - 1)
        if score == float("inf"):
            return (2 ** 31 - 1)
        return score

    # This algorithm will produce an action given a state.
    # Here we use the epsilon-greedy algorithm: with probability
    # |explorationProb|, take a random action.
    def getAction(self, state):
        self.numIters += 1
        if random.random() < self.explorationProb:
            return random.choice([0, 1, 2, 3])
        else:
            return max((self.getQ(state, action), action) for action in self.actions)[1]

    # Call this function to get the step size to update the weights.
    def getStepSize(self):
        return 1.0 / math.sqrt(self.numIters)

    # We will call this function with (s, a, r, s'), which you should use to update |weights|.
    # Note that if s is a terminal state, then s' will be None.  Remember to check for this.
    # You should update the weights using self.getStepSize(); use
    # self.getQ() to compute the current estimate of the parameters.
    def incorporateFeedback(self, state, action, reward, newState):
        
            
        # BEGIN_YOUR_CODE (our solution is 12 lines of code, but don't worry if you deviate from this)
        curFeatures = self.featureExtractor(state, action)
        vopt = float("-inf")
        if newState != None:
            for action in self.actions:
                if self.getQ(newState, action) > vopt:
                    vopt = self.getQ(newState, action)
            #vopt = max([self.getQ(newState, action) for action in self.actions(newState)])
        else:
            vopt = 0
        # print self.weights
        n = self.stepSize
        prediction = self.getQ(state, action)

        target = reward + (self.discount * vopt)

        for key, value in curFeatures:
            if self.weights[key] > (2 ** 31 - 1):
                self.weights[key] = (2 ** 31 - 1)
            elif self.weights[key] < (2 ** 31 - 1):
                self.weights[key] = (2 ** 31 - 1)
            else:
                self.weights[key] -= (n * (prediction - target) * value)
            #self.weights[key] = float('%.3f'%(self.weights[key]))


def featureExtractor(state, action):
    toReturn = []
    toReturn.append(((("pos", state[0], state[1]), action), 1))
    toReturn.append(((("xvel", state[2], ""), action), 1))
    toReturn.append(((("yvel", state[3], "", ""), action), 1))
    toReturn.append(((("angular_v", state[4], "", "", ""), action), 1))
    toReturn.append(((("angular_pos", state[5], "", "", "", ""), action), 1))
    toReturn.append(((("leftleg","", "", state[6]), action), 1))
    toReturn.append(((("rightleg","", "", "", state[7]), action), 1))
    return toReturn

def discretize(state):

    def set_between(obs, low, high):
        if obs > high:
            return 1
        elif obs < low:
            return -1
        return 0

    x_pos = set_between(state[0], -.2, .2)
    y_pos = set_between(state[1], .1, .5)
    x_vel = set_between(state[2], -.05, .05)
    y_vel = set_between(state[3], -.8, -.2)
    angular_v = set_between(state[4], -.1, .1)
    angular_pos = set_between(state[5], -.3, .3)
    return (x_pos, y_pos, x_vel, y_vel, angular_v, angular_pos, state[6], state[7])


env = gym.make('LunarLander-v2')
env.reset()
observation = [0,0,0,0,0,0,0,0]
rl = QLearningAlgorithm([0, 1, 2, 3], 1,
                                       featureExtractor,
                                       0.2)
np.seterr("ignore", "ignore", "ignore", "ignore", "raise")

action = 0
for i in range(900):
    avgReward = 0
    totalReward = 0
    print("iteration{}".format(i))
    env.reset()
    for _ in range(1000):

        #env.render()
        newAction = heuristic(env, observation)
        
        #action = input("what movement?")
        #action = int(action)
        
        newObservation, reward, done, info = env.step(newAction)
        rl.incorporateFeedback(discretize(observation), action, reward, discretize(newObservation))
        totalReward += reward
        posX = newObservation[0] #left, center, or right
        posY = newObservation[1] #left, center, or right
        velX = newObservation[2] #going left, going right
        velY = newObservation[3] #going left, going right
        verticalAngle = observation[4] #towards upright, away from upright, not moving
        angularVelocity = observation[5] #towards upright, away from upright, not moving
        leftLeg = observation[6] #success! or failure
        rightLeg = observation[7] #success! or failure
        #time.sleep(.1)
        action = newAction
        observation = newObservation
        
        if (done):
            avgReward += totalReward
            print("reward:{}".format(totalReward))
            break    
print ("average:{}".format((1.0 * avgReward)/900.0))
print(rl.weights)


action = 0
avgReward = 0
for i in range(1000):
    totalReward = 0
    print("iteration{}".format(i))
    env.reset()
    for _ in range(1000):
        #env.render()
        newAction = rl.getAction(discretize(observation))
        
        #action = input("what movement?")
        #action = int(action)
        
        newObservation, reward, done, info = env.step(newAction)
        rl.incorporateFeedback(discretize(observation), action, reward, discretize(newObservation))
        totalReward += reward
        posX = newObservation[0] #left, center, or right
        posY = newObservation[1] #left, center, or right
        velX = newObservation[2] #going left, going right
        velY = newObservation[3] #going left, going right
        verticalAngle = newObservation[4] #towards upright, away from upright, not moving
        angularVelocity = newObservation[5] #towards upright, away from upright, not moving
        leftLeg = newObservation[6] #success! or failure
        rightLeg = newObservation[7] #success! or failure
        #time.sleep(.1)
        action = newAction
        observation = newObservation
        
        if (done):
            avgReward += totalReward
            print("reward:{}".format(totalReward))
            break    
print ("average:{}".format((1.0 * avgReward)/1000.0))
print(rl.weights)
print("WE LEARNED LETS DO THIS")
observation = [0,0,0,0,0,0,0,0]
avgReward = 0
for k in range(10):
    totalReward = 0
    env.reset()
    print("learned:{}".format(k))
    for _ in range(1000):
        env.render()
        print ("{}".format(_))
        state = discretize(observation)
        newAction = max((rl.getQ(state, action), action) for action in rl.actions)[1]
        print ("newaction:{}".format(newAction))
        
        #action = input("what movement?")
        #action = int(action)
        
        newObservation, reward, done, info = env.step(newAction)
        print("uodating reward:{}".format(totalReward))
        totalReward += reward
        #rl.incorporateFeedback(discretize(observation), action, reward, discretize(observation))
        action = newAction
        observation = newObservation
        if (done):
            avgReward += totalReward
            print("reward:{}".format(totalReward)) 
            break; 

print("average:{}".format((1.0 * avgReward)/10.0))

