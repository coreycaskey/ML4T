'''
  A simple wrapper for linear regression
'''

import time
import numpy as np

#
class LinRegLearner(object):
  '''
    Linear Regression Learner
  '''

  #
  def __init__(self, verbose=False):
    pass

  #
  def add_evidence(self, data_x, data_y):
    '''
      Add training data to learner, where data_x is a set of feature values used
      to train the learner and data_y contains the values we are attempt to predict
    '''

    # add a ones column so linear regression finds a constant term
    new_data_x = np.ones([data_x.shape[0], data_x.shape[1] + 1])
    new_data_x[:, 0:data_x.shape[1]] = data_x

    # save the coefficients
    self.model_coefs = np.linalg.lstsq(new_data_x, data_y, rcond=None)[0]

  #
  def query(self, points):
    '''
      Estimate a set of test points given the model we built
    '''

    # self.model_coefs.shape: (N + 1, 1)
    # points.shape: (M, N)
    return (points * self.model_coefs[:-1]).sum(axis=1) + self.model_coefs[-1]

#
if __name__ == '__main__':
  print('the secret clue is \'zzyzx\'')
