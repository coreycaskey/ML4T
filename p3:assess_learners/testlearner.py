'''
  Test the learning algorithms
'''

import math
import sys

import numpy as np

import LinRegLearner as lrl

if __name__ == '__main__':
  if len(sys.argv) != 2:
    print('Usage: python testlearner.py Data/<filename>')
    sys.exit(1)

  inf = open(sys.argv[1])
  data = np.array([ list(map(float, s.strip().split(','))) for s in inf.readlines() ])

  # compute how much of the data is training and testing
  train_rows = int(0.6 * data.shape[0])
  test_rows = data.shape[0] - train_rows

  # separate out training and testing data
  train_x = data[:train_rows, 0:-1]
  train_y = data[:train_rows, -1]
  test_x = data[train_rows:, 0:-1]
  test_y = data[train_rows:, -1]

  print('Training Data X: {}'.format(train_x.shape))
  print('Training Data Y: {}'.format(train_y.shape))

  print('Testing Data X: {}'.format(test_x.shape))
  print('Testing Data Y: {}'.format(test_y.shape))

  # create a learner and train it
  learner = lrl.LinRegLearner(verbose=True) # create a LinRegLearner
  learner.add_evidence(train_x, train_y) # train it

  # evaluate in-sample
  pred_y = learner.query(train_x)
  rmse = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])

  print()
  print('In-sample results')
  print('RMSE: {}'.format(rmse))

  c = np.corrcoef(pred_y, y=train_y)

  print('Corr: {}'.format(c[0,1]))

  # evaluate out-of-sample
  pred_y = learner.query(test_x)
  rmse = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])

  print()
  print('Out-of-sample results')
  print('RMSE: {}'.format(rmse))

  c = np.corrcoef(pred_y, y=test_y)

  print('Corr: {}'.format(c[0,1]))
