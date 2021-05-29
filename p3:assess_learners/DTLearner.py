'''
  ...
'''

import numpy as np

from scipy.stats.stats import pearsonr

#
class DTLearner(object):
  '''
    Decision Tree Learner
  '''

  #
  def __init__(self, leaf_size=1, verbose=False):
    self.leaf_size = leaf_size

  #
  def add_evidence(self, data_x, data_y):
    '''
      :param data_x: a set of feature values used to train the learner
      :param data_y: the value we are attempting to predict given the X data
    '''

    data = np.column_stack((data_x, data_y)) # pair final column with Xi features

    self.d_tree = self.build_tree(data) # recursive tree building call

  #
  def build_tree(self, data):
    '''
      @param data: training data for building tree
      @returns the leaf of the tree
    '''

    # None value type denotes leaf node
    # otherwise, has feature index value
    if data.shape[0] == 1: # only one instance
      return np.array([[None, data[0, -1], np.nan, np.nan]])

    elif np.unique(data[:, -1]).size == 1:  # all same Y value
      return np.array([[None, data[0, -1], np.nan, np.nan]])

    elif data.shape[0] <= self.leaf_size: # leaf_size or fewer elements
      leaf_split = np.mean(data[:, -1]) # aggregate Y values
      return np.array([[None, leaf_split, np.nan, np.nan]])

    else:
      max_index = self.bestFeature(data)
      split_val = np.median(data[:, max_index])               # median from best feature column values

      left = data[data[:, max_index] <= split_val]            # elements less than/equal to split_val
      right = data[data[:, max_index] > split_val]            # elements greater than split_val
      ds = data.shape[0]

      if left.shape[0] == ds or right.shape[0] == ds:                     # split_val failed to split any further
        leaf_split = np.mean(data[:, -1])                               # aggregate Y values
        return np.array([[None, leaf_split, np.nan, np.nan]])           # make leaf node

      lTree = self.buildTree(left)                                        # make subtrees
      rTree = self.buildTree(right)
      root = np.array([[max_index, split_val, 1, lTree.shape[0] + 1]])

      return np.vstack((root, lTree, rTree))                              # stack tree nodes (rows) to tree (matrix)

  #
  def bestFeature(self, data):
    '''
      @summary: Finds the Xi feature with the highest abs val correlation.
      @param data: training data for building tree.
      @returns the index of best Xi feature.
    '''
    num_xi = data.shape[1] - 1                              # num of Xi features; ignore Y column
    feat_corr = np.zeros(num_xi)                            # list to fill with correlations
    y_vals = data[:, -1]                                    # Y column values

    for i in range(num_xi):                                 # iterate over all features
        curr_feat = data[:, i]                              # values for current Xi feature
        corr_tuple = pearsonr(curr_feat, y_vals)            # get correlation value
        corr_arr = np.absolute(np.asarray(corr_tuple))      # list of absolute value correlations
        feat_corr[i] = corr_arr[0]                          # store correlation

    max_index = np.argmax(feat_corr, axis = None)           # index of best feature (max abs val correlation)

    return max_index

  #
  def query(self, points):
    '''
      Estimate a set of test points given the model we built

      :param points: should be a numpy array with each row corresponding to a specific query.
      :returns the estimated values according to the saved model.
    '''

    pred_y = []

    for point in points:
      leaf_val = self.traverse(0, point) # get predicted Y value (start at root node)
      pred_y.append(leaf_val)

    return np.array(pred_y)

  #
  def traverse(self, i, point):
    '''
      @summary: Recursively traverse decision tree for given training/testing point
      @param i: index of node in decision tree (first call starts at root node)
      @param point: the actual testing/training point from the matrix
      @returns the predicted Y value for the point.
    '''

    node = self.dTree[i]        # info for tree node

    if node[0] is None:         # reached leaf node
      return node[1]          # use node split value

    feat_index = int(node[0])   # feature index for node comparison

    if point[feat_index] <= node[1]:                    # less than/equal to split value
      return self.traverse(i + int(node[2]), point)   # go to left subtree

    else:
      return self.traverse(i + int(node[3]), point)   # go to right subtree

#
if __name__=='__main__':
  print('the secret clue is \'zzyzx\'')
