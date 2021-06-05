"""
    Test the learning algorithms

    Usage:
    - Point the terminal location to this directory
    - Run the command `PYTHONPATH=../:. python testlearner.py data/<filename>`
"""

import math
import sys
import time

import numpy as np
import matplotlib.pyplot as plt

import LinRegLearner as lrl
import DTLearner as dt
import RTLearner as rt
import BagLearner as bl
import InsaneLearner as il

#
if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python testlearner.py data/<filename>')
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

    print()
    print('---------------------------------------------------------')
    print()

    ###################################
    #### Linear Regression Learner ####
    ###################################

    print('Linear Regression Learner')

    learner = lrl.LinRegLearner()
    learner.add_evidence(train_x, train_y)

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
    print()
    print('---------------------------------------------------------')
    print()

    ##################################
    ########### DT Learner ###########
    ##################################

    print('Decision Tree Learner')

    learner = dt.DTLearner()
    learner.add_evidence(train_x, train_y)

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
    print()
    print('---------------------------------------------------------')
    print()

    ##################################
    ########### RT Learner ###########
    ##################################

    print('Random Tree Learner')

    learner = rt.RTLearner()
    learner.add_evidence(train_x, train_y)

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
    print()
    print('---------------------------------------------------------')
    print()

    ###################################
    ########### Bag Learner ###########
    ###################################

    print('Bag Learner')

    learner = bl.BagLearner() # defaults to DTLearner with leaf size of 1
    learner.add_evidence(train_x, train_y)

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
    print()
    print('---------------------------------------------------------')
    print()

    ####################################
    ########## Insane Learner ##########
    ####################################

    print('Insane Learner')

    learner = il.InsaneLearner() # defaults to DTLearner with leaf size of 1
    learner.add_evidence(train_x, train_y)

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
    print()
    print('---------------------------------------------------------')
    print()

    ##############################################
    ################## FIGURE 1 ##################
    ######### OVERFITTING WITH LEAF SIZE #########
    ################ ISTANBUL.CSV ################
    ##############################################

    max_leaf_size = 50
    in_sample = np.zeros(max_leaf_size + 1)
    out_of_sample = np.zeros(max_leaf_size + 1)

    for i in range(max_leaf_size + 1):
        learner = dt.DTLearner(leaf_size=i)
        learner.add_evidence(train_x, train_y)

        # evaluate in sample
        pred_y = learner.query(train_x)
        rmse = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])
        in_sample[i] = rmse

        # evaluate out of sample
        pred_y = learner.query(test_x)
        rmse = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])
        out_of_sample[i] = rmse

    plt.figure(1)
    plt.axis([0, max_leaf_size, 0, 0.01])

    plt.xlabel('Leaf Size')
    plt.ylabel('Root Mean Squared Error (RMSE)')
    plt.title('Figure 1: Overfitting w/ DT Learner Leaf Size')

    plt.plot(in_sample, label='In Sample')
    plt.plot(out_of_sample, label='Out of Sample')

    plt.legend(loc='lower right', shadow=True, fontsize='medium')
    plt.savefig('Figure_1.png')

    ##############################################
    ################## FIGURE 2 ##################
    ########## OVERFITTING WITH BAGGING ##########
    ################ ISTANBUL.CSV ################
    ##############################################

    max_leaf_size = 50
    in_sample = np.zeros(max_leaf_size + 1)
    out_of_sample = np.zeros(max_leaf_size + 1)

    for i in range(max_leaf_size + 1):
        learner = bl.BagLearner(kwargs={'leaf_size':i})
        learner.add_evidence(train_x, train_y)

        # evaluate in sample
        pred_y = learner.query(train_x)
        rmse = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])
        in_sample[i] = rmse

        # evaluate out of sample
        pred_y = learner.query(test_x)
        rmse = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])
        out_of_sample[i] = rmse

    plt.figure(2)
    plt.axis([0, max_leaf_size, 0, 0.01])

    plt.xlabel('Leaf Size')
    plt.ylabel('Root Mean Squared Error (RMSE)')
    plt.title('Figure 2: Overfitting w/ DT Learner Using Bags')

    plt.plot(in_sample, label='In Sample')
    plt.plot(out_of_sample, label='Out of Sample')

    plt.legend(loc='lower right', shadow=True, fontsize='medium')
    plt.savefig('Figure_2.png')

    ##############################################
    ################## FIGURE 3 ##################
    ########### DT vs RT Training Time ###########
    ################ ISTANBUL.CSV ################
    ##############################################

    max_leaf_size = 50
    dt_time = np.zeros(max_leaf_size + 1)
    rt_time = np.zeros(max_leaf_size + 1)

    for i in range(max_leaf_size + 1):
        learner = dt.DTLearner(leaf_size=i)
        start = time.time()
        learner.add_evidence(train_x, train_y)
        end = time.time()
        dt_time[i] = (end - start) * 1000 # time to train

        #############################

        learner = rt.RTLearner(leaf_size=i)
        start = time.time()
        learner.add_evidence(train_x, train_y)
        end = time.time()
        rt_time[i] = (end - start) * 1000 # time to train

    plt.figure(3)
    plt.axis([0, max_leaf_size, 0, 50])

    plt.xlabel('Leaf Size')
    plt.ylabel('Milliseconds')
    plt.title('Figure 3: RT vs. DT Learner Training Runtime')

    plt.plot(dt_time, label='DT Learner')
    plt.plot(rt_time, label='RT Learner')

    plt.legend(loc='upper right', shadow=True, fontsize='medium')
    plt.savefig('Figure_3.png')

    ##############################################
    ################## FIGURE 4 ##################
    ########### DT vs RT Balance Ratio ###########
    ################ ISTANBUL.CSV ################
    ##############################################

    max_leaf_size = 50
    dt_size = np.zeros(max_leaf_size + 1)
    rt_size = np.zeros(max_leaf_size + 1)

    for i in range(0, max_leaf_size + 1):
        dt_learner = dt.DTLearner(leaf_size=i)
        dt_learner.add_evidence(train_x, train_y)

        dt_root = dt_learner.d_tree[0]
        dt_offset = dt_learner.d_tree[1]
        dt_right = 0
        dt_left = 1

        while dt_root[0] is not None:
            dt_right += dt_root[3]
            dt_root = dt_learner.d_tree[int(dt_right)]

        while dt_offset[0] is not None:
            dt_left += dt_offset[3]
            dt_offset = dt_learner.d_tree[int(dt_left)]

        dt_size[i] = (dt_right + 1) / float(dt_left)

        #############################

        rt_learner = rt.RTLearner(leaf_size=i)
        rt_learner.add_evidence(train_x, train_y)

        rt_root = rt_learner.r_tree[0]
        rt_offset = rt_learner.r_tree[1]
        rt_right = 0
        rt_left = 1

        while rt_root[0] is not None:
            rt_right += rt_root[3]
            rt_root = rt_learner.r_tree[int(rt_right)]

        while rt_offset[0] is not None:
            rt_left += rt_offset[3]
            rt_offset = rt_learner.r_tree[int(rt_left)]

        rt_size[i] = (rt_right + 1) / float(rt_left)

    plt.figure(4)
    plt.axis([1, max_leaf_size, 1.5, 2.5])

    plt.xlabel('Leaf Size')
    plt.ylabel('Ratio of Right Subtree to Left Subtree')
    plt.title('Figure 4: RT Learner vs. DT Learner Tree Balance')

    plt.plot(dt_size, label='DT Learner')
    plt.plot(rt_size, label='RT Learner')

    plt.legend(loc='upper right', shadow=True, fontsize='medium')
    plt.savefig('Figure_4.png')

    ##############################################
    ################## FIGURE 5 ##################
    ############### DT vs RT Error ###############
    ################ ISTANBUL.CSV ################
    ##############################################

    max_leaf_size = 50
    dt_rmse_in_sample = np.zeros(max_leaf_size + 1)
    rt_rmse_in_sample = np.zeros(max_leaf_size + 1)
    dt_rmse_out_of_sample = np.zeros(max_leaf_size + 1)
    rt_rmse_out_of_sample = np.zeros(max_leaf_size + 1)

    for i in range(0, max_leaf_size + 1):
        dt_learner = dt.DTLearner(leaf_size=i)
        dt_learner.add_evidence(train_x, train_y)

        # evaluate in sample
        pred_y = dt_learner.query(train_x)
        rmse = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])
        dt_rmse_in_sample[i] = rmse

        # evaluate out of sample
        pred_y = dt_learner.query(test_x)
        rmse = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])
        dt_rmse_out_of_sample[i] = rmse

        #############################

        rt_learner = rt.RTLearner(leaf_size=i)
        rt_learner.add_evidence(train_x, train_y)

        # evaluate in sample
        pred_y = rt_learner.query(train_x)
        rmse = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])
        rt_rmse_in_sample[i] = rmse

        # evaluate out of sample
        pred_y = rt_learner.query(test_x) # get the predictions
        rmse = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])
        rt_rmse_out_of_sample[i] = rmse

    plt.figure(5)
    plt.axis([1, max_leaf_size, 0, 0.01])

    plt.xlabel('Leaf Size')
    plt.ylabel('Root Mean Squared Error (RMSE)')
    plt.title('Figure 5: RT Learner vs. DT Learner RMSE')

    plt.plot(dt_rmse_in_sample, label='DT In Sample')
    plt.plot(dt_rmse_out_of_sample, label='DT Out of Sample')
    plt.plot(rt_rmse_in_sample, label='RT In Sample')
    plt.plot(rt_rmse_out_of_sample, label='RT Out of Sample')

    plt.legend(loc='lower right', shadow=True, fontsize='medium')

    plt.savefig('Figure_5.png')
