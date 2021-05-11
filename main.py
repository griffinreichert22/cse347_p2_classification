##########################
# Griffin Reichert
# CSE 347 Final Project
#########################

# use conda activate tf to run in anaconda environment

############
# imports
############
import time
import numpy as np
# import tensorflow as tf
from tensorflow.keras import datasets

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, accuracy_score

import xgboost as xgb
import matplotlib.pyplot as plt



# # main program

# ##########################
# # Load data
# ##########################
print('\nloading data...')

# Get MNIST data
(m_train, m_train_y), (m_test, m_test_y) = datasets.mnist.load_data() 
print(f'loaded mnist: {m_train.shape} {m_test.shape}')
m_train = np.reshape(m_train, (m_train.shape[0], m_train.shape[1]*m_train.shape[2]))
m_test = np.reshape(m_test, (m_test.shape[0], m_test.shape[1]*m_test.shape[2]))
print(f'reshaped mnist: {m_train.shape} {m_test.shape}')

# Load PIE data
p_train = np.loadtxt('./data/PIE/StTrainFile1.txt')
p_test = np.loadtxt('./data/PIE/StTestFile1.txt')
p_train_y = p_train[:, -1] - 1
p_train = p_train[:, :-1]
p_test_y = p_test[:, -1] - 1
p_test = p_test[:, :-1]
print(f'loaded PIE: {p_train.shape} {p_train_y.shape}')

# Load YaleB data
yb_train = np.loadtxt('./data/YaleB/StTrainFile1.txt')
yb_test = np.loadtxt('./data/YaleB/StTestFile1.txt')
yb_train_y = yb_train[:, -1] - 1
yb_train = yb_train[:, :-1]
yb_test_y = yb_test[:, -1] - 1
yb_test = yb_test[:, :-1]
print(f'loaded YaleB: {yb_train.shape}')

# Get CIFAR data
# (c_train, c_train_y), (c_test, c_test_y) = datasets.cifar10.load_data() 
# print(f'loaded cifar: {c_train.shape} {c_test.shape}')
# c_train_y = np.reshape(c_train_y, c_train_y.shape[0]) # cifar labels are weird: [[6], [4]] convert to [6, 4]
# c_test_y = np.reshape(c_test_y, c_test_y.shape[0])
# c_train = np.reshape(c_train, (c_train.shape[0], c_train.shape[1]*c_train.shape[2]*c_train.shape[3]))
# c_test = np.reshape(c_test, (c_test.shape[0], c_test.shape[1]*c_test.shape[2]*c_test.shape[3]))
# print(f'reshaped cifar: {c_train.shape} {c_test.shape}')

#####################
# XGBoost
#####################
sc = StandardScaler()

# TODO could refactor scaling to be done in importing data section

a = time.time()

# dtrain = xgb.DMatrix(sc.fit_transform(yb_train), label=yb_train_y)
# dtest = xgb.DMatrix(sc.fit_transform(yb_test), label=yb_test_y)
# # read about paramerters here: https://xgboost.readthedocs.io/en/latest/parameter.html
# param_list = { 
#     "eta": 0.08, 
#     "subsample": 0.75, 
#     "colsample_bytree": 0.8, 
#     "objective": "multi:softmax", 
#     "eval_metric": "merror", 
#     "alpha": 8, 
#     "lambda": 3, 
#     "num_class": 38
# }
# YaleB accuracy: 0.7763  duration: 38.92
# dtrain = xgb.DMatrix(sc.fit_transform(p_train), label=p_train_y)
# dtest = xgb.DMatrix(sc.fit_transform(p_test), label=p_test_y)
# pie_param_list = { 
#     "eta": 0.08, 
#     "subsample": 0.8, 
#     "colsample_bytree": 0.8, 
#     "objective": "multi:softmax", 
#     "eval_metric": "merror", 
#     "alpha": 2, 
#     "lambda": 2, 
#     "num_class": 68
# }

dtrain = xgb.DMatrix(sc.fit_transform(m_train), label=m_train_y)
dtest = xgb.DMatrix(sc.fit_transform(m_test), label=m_test_y)
param_list = { 
    "eta": 0.08, 
    "subsample": 0.8, 
    "colsample_bytree": 0.8, 
    "objective": "multi:softmax", 
    "eval_metric": "merror", 
    "alpha": 8, 
    "lambda": 2, 
    "num_class": 10
}
# MNIST accuracy: 0.9611  duration: 1735.71

evallist = [(dtest, 'eval'), (dtrain, 'train')]
n_rounds = 300
early_stopping = 50

bst = xgb.train(param_list, dtrain, n_rounds, evals=evallist, early_stopping_rounds=early_stopping, verbose_eval=True)
y_pred = bst.predict(dtest)
b = time.time()

m_acc = accuracy_score(m_test_y, y_pred)
print('MNIST accuracy: {:.4f}  duration: {:.2f}'.format(m_acc, (b-a))) 

######################
# SVM
######################

# print('\nrunning svm on YaleB...')
# a = time.time()
# yb_svm = make_pipeline(StandardScaler(), SVC(kernel='linear')) # YaleB accuracy: 0.9518  duration: 3.21
# # yb_svm = SVC(kernel='rbf') # accuracy: 0.8509  duration: 7.55
# yb_svm.fit(yb_train, yb_train_y)
# yb_acc = yb_svm.score(yb_test, yb_test_y)
# b = time.time()
# print('YaleB accuracy: {:.4f}  duration: {:.2f}'.format(yb_acc, (b-a))) 

# print('\nrunning svm on PIE...')
# a = time.time()
# p_svm = make_pipeline(StandardScaler(), SVC(kernel='linear')) # PIE accuracy: 0.6883  duration: 35.51
# # p_svm = SVC(kernel='linear') # acc 0.6718  duration: 37.5 sec
# # p_svm = SVC(kernel='rbf') # acc 0.3472 in 94 sec
# # p_svm = LinearSVC(random_state=0) # acc 0.6727 in 42 sec
# p_svm.fit(p_train, p_train_y)
# p_acc = p_svm.score(p_test, p_test_y)
# b = time.time()
# print('PIE accuracy: {:.4f}  duration: {:.2f}'.format(p_acc, (b-a)))

# print('\nrunning svm on mnist...')
# a = time.time()
# m_svm = make_pipeline(StandardScaler(), SVC(kernel='rbf')) # mnist accuracy: 0.9660  duration: 791.99,  (with linear): 0.9293  duration: 649.43
# # m_svm = SVC(kernel='rbf') # mnist accuracy: 0.9792  duration: 506 sec
# # m_svm = LinearSVC(random_state=0) # acc 0.8783 in 2:09 min 129 sec
# m_svm.fit(m_train, m_train_y)
# m_acc = m_svm.score(m_test, m_test_y)
# b = time.time()
# print('mnist accuracy: {:.4f}  duration: {:.2f}'.format(m_acc, (b-a)))

# print('\nrunning svm on cifar...')
# a = time.time()
# c_svm = LinearSVC(random_state=0)
# c_svm.fit(c_train, c_train_y)
# c_acc = c_svm.score(c_test, c_test_y)
# b = time.time()
# print(f'cifar accuracy: {c_acc}  duration: {b-a}') # best acc 0.2594 in 56 min



#####################
# Main Program 
#####################

# print("\nWelcome to my classification program!")
# print("  [1] XGBoost")
# print("  [2] Support Vector Machines")
# print("  [3] Convolutional Neural Network")
# alg = input("Please enter the number of the algorithm you wish to use (q to quit): ")
# while alg != 'q':
#     if alg == '1':
#         print('todo xgb')
#     elif alg == '2':
#         print('todo svm')
#     elif alg == '3':
#         print('todo cnn')
#     else:
#         alg = input("Only enter 1, 2, or 3 (q to quit): ")
#         continue
#     break
    # alg = input("Please enter the number of the algorithm you wish to use (q to quit): ")



##########################
# Links 
##########################
# https://towardsdatascience.com/image-classification-in-10-minutes-with-mnist-dataset-54c35b77a38d
