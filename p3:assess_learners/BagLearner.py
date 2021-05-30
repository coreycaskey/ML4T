'''
  A simple wrapper for Bootstrap Aggregation Learner (Bag Learner)
'''

import numpy as np
import DTLearner as dt
#
class BagLearner(object):
  '''
    Bootstrap Aggregation Learner
  '''

  #
  def __init__(self, learner=dt.DTLearner, kwargs={'leaf_size':1}, bags=20, boost=False, verbose=False):
    learners = []

    for _ in range(bags):
      learners.append(learner(**kwargs))

    self.learners = learners

  #
  def add_evidence(self, data_x, data_y):
    '''
      Randomly select a subset of training data from data_x and data_y for the bag that
      a learner will use to train their model
    '''

    data = np.column_stack((data_x, data_y))
    n = data.shape[0] # amount of training data

    for learner in self.learners:
      bag = np.empty(shape=(0, data.shape[1]))

      for _ in range(n):
        index = np.random.randint(0, n) # sample with replacement
        bag = np.vstack((bag, data[index]))

      bag_x = bag[:, 0:-1]
      bag_y = bag[:, -1]

      learner.add_evidence(bag_x, bag_y)

  #
  def query(self, points):
    '''
      Estimate a set of test points given the model we built (i.e. average the different
      results of the bagging outputs)
    '''

    bag_outputs = []

    for learner in self.learners:
      pred_y = learner.query(points)
      bag_outputs.append(pred_y)

    return np.mean(np.array(bag_outputs), axis=0)

#
if __name__=='__main__':
  print('the secret clue is \'zzyzx\'')
