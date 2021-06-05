"""
    A simple wrapper for Insane Learner
"""

import numpy as np
import BagLearner as bl
import LinRegLearner as lrl

class InsaneLearner(object):
    """
        Insane Learner
    """

    def __init__(self, verbose=False):
        self.learners = []

        for _ in range(20):
            self.learners.append(bl.BagLearner(learner=lrl.LinRegLearner, kwargs={}))

    #
    def add_evidence(self, data_x, data_y):
        """
            Pass the training data_x and data_y to the bag learner
        """

        for learner in self.learners:
            learner.add_evidence(data_x, data_y)

    #
    def query(self, points):
        """
            Pass the testing points to the bag learner
        """

        bag_outputs = []

        for learner in self.learners:
            bag_outputs.append(learner.query(points))

        return np.mean(np.array(bag_outputs), axis=0)
