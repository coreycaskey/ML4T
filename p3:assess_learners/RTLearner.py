'''
  A simple wrapper for Random Trees
'''

import numpy as np

#
class RTLearner(object):
  '''
    Random Tree Learner
  '''

  #
  def __init__(self, leaf_size=1, verbose=False):
    self.leaf_size = leaf_size

  #
  def add_evidence(self, data_x, data_y):
    '''
      Add training data to learner, where data_x is a set of feature values used
      to train the learner and data_y contains the values we are attempt to predict
    '''

    data = np.column_stack((data_x, data_y))

    self.r_tree = self.build_tree(data)

  #
  def build_tree(self, data):
    '''
      Recursively builds the tree with the training data and returns the leaf
      of the tree
    '''

    # A random tree node (row) in tabular form has the following structure:
    #   - [ feature index, split value, left node (index offset from current node), right node (index offset from current node) ]
    #
    # A leaf node (row) has a slightly different structure:
    #   - [ None (i.e. not a feature), Y value, NaN, NaN ]

    if data.shape[0] == 1: # one instance of data
      return np.array([[None, data[0, -1], np.nan, np.nan]])

    elif np.unique(data[:, -1]).size == 1: # all same Y value
      return np.array([[None, data[0, -1], np.nan, np.nan]])

    elif data.shape[0] <= self.leaf_size: # leaf_size or fewer elements (aggregate data into leaf)
      return np.array([[None, np.mean(data[:, -1]), np.nan, np.nan]])

    else:
      rand_feature_index = np.random.randint(0, data.shape[1] - 1)
      split_val = np.median(data[:, rand_feature_index])

      left_data = data[data[:, rand_feature_index] <= split_val] # feature values less than / equal to the split value
      right_data = data[data[:, rand_feature_index] > split_val] # feature values greater than the split value

      # failed to split any further (aggregate data into leaf)
      if left_data.shape[0] == data.shape[0] or right_data.shape[0] == data.shape[0]:
        return np.array([[None, np.mean(data[:, -1]), np.nan, np.nan]])

      left_tree = self.build_tree(left_data)
      right_tree = self.build_tree(right_data)

      root_node = np.array([[rand_feature_index, split_val, 1, left_tree.shape[0] + 1]])

      return np.vstack((root_node, left_tree, right_tree)) # stack tree nodes (rows) in tree matrix

  #
  def query(self, points):
    '''
      Estimate a set of test points given the model we built
    '''

    pred_y = []

    for point in points:
      leaf_val = self.traverse(0, point) # get predicted Y value
      pred_y.append(leaf_val)

    return np.array(pred_y)

  #
  def traverse(self, node_index, point):
    '''
      Recursively traverses the random tree based on a given training / testing
      point, where the row index in the RT matrix is used to identify the current node
    '''

    node = self.r_tree[node_index]

    if node[0] is None: # reached leaf node
      return node[1]

    feature_index = int(node[0])

    if point[feature_index] <= node[1]: # value of point at feature index is less than / equal to the split value
      return self.traverse(node_index + int(node[2]), point)

    else:
      return self.traverse(node_index + int(node[3]), point)
