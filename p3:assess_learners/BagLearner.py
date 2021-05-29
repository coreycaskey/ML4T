'''
A simple wrapper for bootstrap aggregation learner.  (c) 2015 Tucker Balch

Copyright 2018, Georgia Institute of Technology (Georgia Tech)
Atlanta, Georgia 30332
All Rights Reserved

Template code for CS 4646/7646

Georgia Tech asserts copyright ownership of this template and all derivative
works, including solutions to the projects assigned in this course. Students
and other users of this template code are advised not to share it with others
or to make it available on publicly viewable websites including repositories
such as github and gitlab.  This copyright statement should not be removed
or edited.

We do grant permission to share solutions privately with non-students such
as potential employers. However, sharing with other current or future
students of CS 7646 is prohibited and subject to being investigated as a
GT honor code violation.

-----do not edit anything above this line---
'''

import numpy as np
import DTLearner as dt

class BagLearner(object):

    def __init__(self, learner = dt.DTLearner, kwargs = {'leaf_size':1}, bags = 20, boost = False, verbose = False):
        learners = []

        for i in range(bags):
            learners.append(learner(**kwargs))

        self.learners = learners

    def author(self):
        return 'ccaskey6'

    def addEvidence(self, dataX, dataY):
        '''
        @summary: Add training data to learner
        @param dataX: X values of data to add
        @param dataY: the Y training values
        '''
        data = np.column_stack((dataX, dataY))          # pair final column with X values
        n = data.shape[0]                               # size of training data

        for learner in self.learners:                   # iterate over all learner instances
            bag = np.empty(shape = (0, data.shape[1]))

            for i in range(n):                          # number of elements to pick
                index = np.random.randint(0, n)         # sample with replacement
                bag = np.vstack((bag, data[index]))

            bagX = bag[:, 0:-1]
            bagY = bag[:, -1]

            learner.addEvidence(bagX, bagY)

    def query(self, points):
        '''
        @summary: Estimate a set of test points given the model we built.
        @param points: should be a numpy array with each row corresponding to a specific query.
        @returns the estimated values according to the saved model.
        '''
        bag_outputs = []

        for learner in self.learners:
            predY = learner.query(points)
            bag_outputs.append(predY)

        bag_outputs = np.array(bag_outputs)

        return np.mean(bag_outputs, axis = 0)

if __name__=='__main__':
    print 'the secret clue is 'zzyzx''
