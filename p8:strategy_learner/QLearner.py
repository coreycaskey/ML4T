"""
    Q-Learning Robot
"""

import numpy as np
import random as rand

#
class QLearner(object):

    #
    def __init__(self, num_states=100, num_actions = 4, alpha = 0.2, gamma = 0.9, rar = 0.5, radr = 0.99, dyna = 0, verbose = False):
        self.verbose = verbose
        self.num_actions = num_actions
        self.s = 0
        self.a = 0

        self.rar = rar
        self.radr = radr
        self.alpha = alpha
        self.gamma = gamma
        self.dyna = dyna

        self.q_table = np.zeros((num_states, num_actions)) # 2D array holding Q values
        self.experience_table = [] # store experience tuples

    #
    def querysetstate(self, s):
        """
            Update the state without updating the Q-table
        """

        self.s = s # set new state
        action = rand.randint(0, self.num_actions-1) if rand.random() < self.rar else np.argmax(self.q_table[s]) # choose random action or action with best Q value

        if self.verbose:
            print('s =', s, 'a =', action)

        return action

    #
    def query(self, s_prime, r):
        """
            Updates the Q-table and return an action
        """

        self.q_table[self.s, self.a] = self.getnewq(self.s, self.a, s_prime, r)
        self.experience_table.append((self.s, self.a, s_prime, r))

        if self.dyna != 0:
            for _ in range(self.dyna):
                rand_exp = self.experience_table[rand.randint(0, len(self.experience_table)-1)] # randomly select experience tuple
                dyna_s, dyna_a, dyna_s_prime, dyna_r = rand_exp[0], rand_exp[1], rand_exp[2], rand_exp[3]

                self.q_table[dyna_s, dyna_a] = self.getnewq(dyna_s, dyna_a, dyna_s_prime, dyna_r)

        action = rand.randint(0, self.num_actions-1) if rand.random() < self.rar else np.argmax(self.q_table[s_prime]) # choose random or best action

        self.rar *= self.radr
        self.s = s_prime
        self.a = action

        if self.verbose:
            print('s =', s_prime, 'a =', action, 'r =', r)

        return action

    #
    def getnewq(self, s, a, s_prime, r):
        """
            Helper function that returns the new Q value for the current state and action
        """

        best_action = np.argmax(self.q_table[s_prime])
        curr_q = self.q_table[s, a]
        max_q = self.q_table[s_prime, best_action]

        return (1 - self.alpha) * curr_q + self.alpha * (r + self.gamma * max_q)

#
if __name__=="__main__":
    print("Remember Q from Star Trek? Well, this isn't him")
