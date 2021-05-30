'''
  Template for generating data to fool learners
'''

import numpy as np
import math

#
def best_for_lin_reg(seed=1489683273):
  np.random.seed(seed)
  rand_gen = np.random.random(size=(100,)) * 200 - 100

  x = []
  y = []

  for row in range(100):
    x.append([rand_gen[row]] * 2)
    y.append(rand_gen[row])

  return np.array(x), np.array(y)

#
def best_for_decision_tree(seed=1489683273):
  np.random.seed(seed)
  rand_gen = np.random.random(size=(100,)) * 200 - 100

  x = []
  y = []

  for row in range(100):
    x.append([rand_gen[row]] * 2)
    y.append(rand_gen[row] ** 2)

  return np.array(x), np.array(y)

#
if __name__=='__main__':
  print('They call me Tim.')
