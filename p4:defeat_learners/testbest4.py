'''
  Test best4 data generator
'''

import numpy as np
import math
import LinRegLearner as lrl
import DTLearner as dt
from gen_data import best_for_lin_reg, best_for_decision_tree

#
def compare_rmse_out_of_sample(learner_1, learner_2, x, y):
  # compute how much of the data is training and testing
  train_rows = int(math.floor(0.6 * x.shape[0]))

  # separate out training and testing data
  train = np.random.choice(x.shape[0], size=train_rows, replace=False)
  test = np.setdiff1d(np.array(range(x.shape[0])), train)

  train_x = x[train, :]
  train_y = y[train]
  test_x = x[test, :]
  test_y = y[test]

  # train the learners
  learner_1.add_evidence(train_x, train_y)
  learner_2.add_evidence(train_x, train_y)

  # evaluate learner_1 out of sample
  pred_y = learner_1.query(test_x)
  rmse_1 = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])

  # evaluate learner_2 out of sample
  pred_y = learner_2.query(test_x)
  rmse_2 = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])

  return rmse_1, rmse_2

#
def test_code():
  lr_learner = lrl.LinRegLearner()
  dt_learner = dt.DTLearner(leaf_size=1)
  x, y = best_for_lin_reg()

  rmse_lr, rmse_dt = compare_rmse_out_of_sample(lr_learner, dt_learner, x, y) # compare the two learners

  print()
  print('best_for_lin_reg() results')
  print('RMSE LR:', rmse_lr)
  print('RMSE DT:', rmse_dt)

  if rmse_lr < 0.9 * rmse_dt:
    print('LR < 0.9 DT: pass')

  else:
    print('LR >= 0.9 DT: fail')

  print()

  ################################

  lr_learner = lrl.LinRegLearner()
  dt_learner = dt.DTLearner(leaf_size=1)
  x, y = best_for_decision_tree()

  rmse_lr, rmse_dt = compare_rmse_out_of_sample(lr_learner, dt_learner, x, y) # compare the two learners

  print()
  print('best_for_decision_tree() results')
  print('RMSE LR:', rmse_lr)
  print('RMSE DT:', rmse_dt)

  if rmse_dt < 0.9 * rmse_lr:
    print('DT < 0.9 LR: pass')

  else:
    print('DT >= 0.9 LR: fail')

  print()

#
if __name__=='__main__':
  test_code()
