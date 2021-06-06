"""
    Test a Q-Learner in a navigation problem

    Usage:
    - Point the terminal location to this directory
    - Run the command `PYTHONPATH=../:. python testqlearner.py`
"""

import numpy as np
import random as rand
import time
import math
import QLearner as ql

#
def printmap(data):
    print('--------------------')

    for row in range(0, data.shape[0]):
        for col in range(0, data.shape[1]):
            if data[row, col] == 0: # Empty space
                print(" ", end=" ")

            if data[row, col] == 1: # Obstacle
                print("O", end=" ")

            if data[row, col] == 2: # El roboto
                print("*", end=" ")

            if data[row, col] == 3: # Goal
                print("X", end=" ")

            if data[row, col] == 4: # Trail
                print(".", end=" ")

            if data[row, col] == 5: # Quick sand
                print("~", end=" ")

            if data[row, col] == 6: # Stepped in quicksand
                print("@", end=" ")

        print()
    print('--------------------')

#
def getrobotpos(data):
    r = -999
    c = -999

    for row in range(0, data.shape[0]):
        for col in range(0, data.shape[1]):
            if data[row,col] == 2:
                c = col
                r = row

    if (r + c) < 0:
        print('warning: start location not defined')

    return r, c

#
def getgoalpos(data):
    r = -999
    c = -999

    for row in range(0, data.shape[0]):
        for col in range(0, data.shape[1]):
            if data[row,col] == 3:
                c = col
                r = row

    if (r + c) < 0:
        print('warning: goal location not defined')

    return (r, c)

#
def movebot(data, oldpos, a):
    testr, testc = oldpos

    randomrate = 0.20 # how often we move randomly
    quicksandreward = -100 # penalty for stepping on quicksand

    if rand.uniform(0.0, 1.0) <= randomrate:
        a = rand.randint(0, 3)

    if a == 0: # north
        testr = testr - 1

    elif a == 1: # east
        testc = testc + 1

    elif a == 2: # south
        testr = testr + 1

    elif a == 3: # west
        testc = testc - 1

    reward = -1 # default reward

    if testr < 0: # off the map
        testr, testc = oldpos

    elif testr >= data.shape[0]: # off the map
        testr, testc = oldpos

    elif testc < 0: # off the map
        testr, testc = oldpos

    elif testc >= data.shape[1]: # off the map
        testr, testc = oldpos

    elif data[testr, testc] == 1: # obstacle
        testr, testc = oldpos

    elif data[testr, testc] == 5 or data[testr, testc] == 6: # quicksand
        reward = quicksandreward
        data[testr, testc] = 6

    elif data[testr, testc] == 3: # goal
        reward = 1

    return (testr, testc), reward

#
def discretize(pos):
    return pos[0] * 10 + pos[1] # convert the location to a single integer

#
def test(map, epochs, learner, verbose):
    # each epoch involves one trip to the goal
    startpos = getrobotpos(map)
    goalpos = getgoalpos(map)
    scores = np.zeros((epochs, 1))

    for epoch in range(1, epochs + 1):
        total_reward = 0
        data = map.copy()
        robopos = startpos
        state = discretize(robopos)
        action = learner.querysetstate(state) # set the state and get first action
        count = 0

        while (robopos != goalpos) & (count<10000):
            # move to new location according to action and then get a new action
            newpos, stepreward = movebot(data, robopos, action)

            if newpos == goalpos:
                r = 1 # reward for reaching the goal

            else:
                r = stepreward # negative reward for not being at the goal

            state = discretize(newpos)
            action = learner.query(state,r)

            if data[robopos] != 6:
                data[robopos] = 4

            if data[newpos] != 6:
                data[newpos] = 2

            robopos = newpos
            total_reward += stepreward
            count = count + 1

        if count == 100000:
            print('timeout')

        if verbose:
            printmap(data)
            print(epoch, total_reward)

        scores[epoch - 1, 0] = total_reward

    return np.median(scores)

#
def test_code():
    verbose = True

    filename = 'testworlds/world01.csv'
    inf = open(filename)
    data = np.array([list(map(float, s.strip().split(","))) for s in inf.readlines()])
    originalmap = data.copy()

    printmap(data)

    rand.seed(5)

    ######## run non-dyna test ########
    learner = ql.QLearner(num_states=100, num_actions = 4, alpha = 0.2, gamma = 0.9, rar = 0.98, radr = 0.999, dyna = 0, verbose=False)
    epochs = 500
    total_reward = test(data, epochs, learner, verbose)

    print(epochs, "median total_reward" , total_reward)
    print()

    non_dyna_score = total_reward

    ######## run dyna test ########
    learner = ql.QLearner(num_states=100, num_actions = 4, alpha = 0.2, gamma = 0.9, rar = 0.5, radr = 0.99, dyna = 200, verbose=False)

    epochs = 50
    data = originalmap.copy()
    total_reward = test(data, epochs, learner, verbose)

    print(epochs, "median total_reward" , total_reward)

    dyna_score = total_reward

    print()
    print()
    print('results for', filename)
    print('non_dyna_score:', non_dyna_score)
    print('dyna_score    :', dyna_score)

#
if __name__=="__main__":
    test_code()
