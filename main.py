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
import tensorflow as tf
from tensorflow.keras import datasets
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelBinarizer
# from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_auc_score

import xgboost as xgb
import matplotlib.pyplot as plt



# # main program





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

# a = time.time()

# dtrain = xgb.DMatrix(yb_train, label=yb_train_y)
# dtest = xgb.DMatrix(yb_test, label=yb_test_y)
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


# dtrain = xgb.DMatrix(p_train, label=p_train_y)
# dtest = xgb.DMatrix(p_test, label=p_test_y)
# param_list = { 
#     "eta": 0.08, 
#     "subsample": 0.8, 
#     "colsample_bytree": 0.8, 
#     "objective": "multi:softmax", 
#     "eval_metric": "merror", 
#     "alpha": 2, 
#     "lambda": 2, 
#     "num_class": 68
# }

# dtrain = xgb.DMatrix(m_train, label=m_train_y)
# dtest = xgb.DMatrix(m_test, label=m_test_y)
# param_list = { 
#     "eta": 0.08, 
#     "subsample": 0.8, 
#     "colsample_bytree": 0.8, 
#     "objective": "multi:softmax", 
#     "eval_metric": "merror", 
#     "alpha": 8, 
#     "lambda": 2, 
#     "num_class": 10
# }
# # MNIST accuracy: 0.9611  duration: 1735.71 ~29 min

# evallist = [(dtest, 'eval'), (dtrain, 'train')]
# n_rounds = 300
# early_stopping = 25
# bst = xgb.train(param_list, dtrain, n_rounds, evals=evallist, early_stopping_rounds=early_stopping, verbose_eval=True)
# y_pred = bst.predict(dtest)
# b = time.time()

# acc = accuracy_score(m_test_y, y_pred)
# print('accuracy: {:.4f}  duration: {:.2f}'.format(acc, (b-a))) 

######################
# SVM
######################

# print('\nrunning svm on YaleB...')
# a = time.time()
# yb_svm = SVC(kernel='linear') # YaleB accuracy: 0.9518  duration: 3.21, YaleB accuracy: 0.9254  duration: 3.17
# # yb_svm = SVC(kernel='rbf') # accuracy: 0.8509  duration: 7.55
# yb_svm.fit(yb_train, yb_train_y)
# yb_acc = yb_svm.score(yb_test, yb_test_y)
# b = time.time()
# print('YaleB accuracy: {:.4f}  duration: {:.2f}'.format(yb_acc, (b-a))) 

# print('\nrunning svm on PIE...')
# a = time.time()
# p_svm = SVC(kernel='linear') # PIE accuracy: 0.6883  duration: 35.51
# # p_svm = SVC(kernel='linear') # acc 0.6718  duration: 37.5 sec
# # p_svm = SVC(kernel='rbf') # acc 0.3472 in 94 sec
# # p_svm = LinearSVC(random_state=0) # acc 0.6727 in 42 sec
# p_svm.fit(p_train, p_train_y)
# p_acc = p_svm.score(p_test, p_test_y)
# b = time.time()
# print('PIE accuracy: {:.4f}  duration: {:.2f}'.format(p_acc, (b-a)))

# print('\nrunning svm on mnist...')
# a = time.time()
# m_svm = SVC(kernel='rbf') # mnist accuracy: 0.9660  duration: 791.99,  (with linear): 0.9293  duration: 649.43
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
# Convolutional NNs 
#####################

# print('\nrunning cnn...')
# ## CNN model from Lecture Examples
# model = models.Sequential()
# model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(16, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.summary()
# ## Add two more dense layers to perform classification
# model.add(layers.Flatten())
# model.add(layers.Dense(32, activation='relu'))
# model.add(layers.Dense(10, activation= 'softmax')) # CIFAR has 10 output classes, so we use a final Dense layer with 10 outputs.
# model.summary()

# yb_train = np.reshape(yb_train, (-1, 32, 32, 1))
# yb_model = Sequential()
# yb_model.add(Conv2D(64, (3,3), input_shape=(32, 32, 1), activation='relu'))
# yb_model.add(MaxPooling2D(pool_size=(2,2)))
# yb_model.add(Conv2D(64, (3,3), activation='relu'))
# yb_model.add(MaxPooling2D(pool_size=(2,2)))
# yb_model.add(Flatten())
# yb_model.add(Dense(64, activation='relu'))
# yb_model.add(Dense(38, activation='softmax')) # 38 classes in YaleB data
# yb_model.compile(optimizer='adam',
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#               metrics=['accuracy'])
# train, val, train_y, val_y = train_test_split(yb_train, yb_train_y, test_size=0.2)
# yb_model.fit(train, train_y, batch_size=64, epochs=30, validation_data=(val, val_y))

# p_train = np.reshape(p_train, (-1, 32, 32, 1))
# p_model = Sequential()
# p_model.add(Conv2D(64, (5,5), input_shape=(32, 32, 1), activation='relu'))
# p_model.add(MaxPooling2D(pool_size=(3,3)))
# p_model.add(Conv2D(32, (3,3), activation='relu'))
# p_model.add(MaxPooling2D(pool_size=(2,2)))
# p_model.add(Flatten())
# p_model.add(Dense(64, activation='relu'))
# p_model.add(Dense(68, activation='softmax')) # 68 classes in pie data
# p_model.compile(optimizer='adam',
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#               metrics=['accuracy'])
# # train, val, train_y, val_y = train_test_split(p_train, p_train_y, test_size=0.2)
# # p_model.fit(train, train_y, batch_size=64, epochs=5, validation_data=(val, val_y)) #accuracy: 0.2961
# p_model.fit(p_train, p_train_y, batch_size=64, epochs=10, validation_data=(np.reshape(p_test, (-1, 32, 32, 1)), p_test_y))

# m_train = np.reshape(m_train, (-1, 28, 28, 1))
# m_model = Sequential()
# m_model.add(Conv2D(64, (3,3), input_shape=(28, 28, 1), activation='relu'))
# m_model.add(MaxPooling2D(pool_size=(2,2)))
# m_model.add(Conv2D(64, (3,3), activation='relu'))
# m_model.add(MaxPooling2D(pool_size=(2,2)))
# m_model.add(Flatten())
# m_model.add(Dense(64, activation='relu'))
# m_model.add(Dense(10, activation='softmax')) # 10 classes in mnist data
# m_model.compile(optimizer='adam',
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#               metrics=['accuracy'])
# train, val, train_y, val_y = train_test_split(m_train, m_train_y, test_size=0.2)
# m_model.fit(train, train_y, batch_size=64, epochs=10, validation_data=(val, val_y))

'''
Epoch 1/10
48000/48000 [==============================] - 36s 752us/sample - loss: 1.6171 - accuracy: 0.8470 - val_loss: 1.5880 - val_accuracy: 0.8722
Epoch 2/10
48000/48000 [==============================] - 38s 802us/sample - loss: 1.5256 - accuracy: 0.9361 - val_loss: 1.4878 - val_accuracy: 0.9735
Epoch 3/10
48000/48000 [==============================] - 42s 870us/sample - loss: 1.4818 - accuracy: 0.9799 - val_loss: 1.4862 - val_accuracy: 0.9754
Epoch 4/10
48000/48000 [==============================] - 42s 868us/sample - loss: 1.4776 - accuracy: 0.9840 - val_loss: 1.4806 - val_accuracy: 0.9808
Epoch 5/10
48000/48000 [==============================] - 41s 848us/sample - loss: 1.4764 - accuracy: 0.9847 - val_loss: 1.4824 - val_accuracy: 0.9787
Epoch 6/10
48000/48000 [==============================] - 43s 892us/sample - loss: 1.4748 - accuracy: 0.9861 - val_loss: 1.4774 - val_accuracy: 0.9837
Epoch 7/10
48000/48000 [==============================] - 41s 853us/sample - loss: 1.4727 - accuracy: 0.9886 - val_loss: 1.4771 - val_accuracy: 0.9838
Epoch 8/10
48000/48000 [==============================] - 41s 844us/sample - loss: 1.4729 - accuracy: 0.9884 - val_loss: 1.4766 - val_accuracy: 0.9846
Epoch 9/10
48000/48000 [==============================] - 42s 884us/sample - loss: 1.4721 - accuracy: 0.9891 - val_loss: 1.4779 - val_accuracy: 0.9830
Epoch 10/10
48000/48000 [==============================] - 41s 862us/sample - loss: 1.4706 - accuracy: 0.9905 - val_loss: 1.4777 - val_accuracy: 0.9833
'''
def eval_performance(y_test, y_pred):
    metrics = {}
    metrics['acc'] = accuracy_score(y_test, y_pred)
    metrics['f1'] = f1_score(y_test, y_pred, average='macro')
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)
    metrics['auc'] = roc_auc_score(y_test, y_pred, average='macro')
    return metrics

def print_metrics(m):
    print(m)


def run_svm(train, train_y, test, ker='linear'):
    svm = SVC(kernel=ker) 
    svm.fit(train, train_y)
    return svm.predict(test)

#####################
# Main Program 
#####################
def main():
    # ##########################
    # # Load data
    # ##########################
    datasplit = input('\nChoose a data split (1-10): ')
    print('\nloading data...')
    sc = StandardScaler()

    # Get MNIST data
    (m_train, m_train_y), (m_test, m_test_y) = datasets.mnist.load_data() 
    m_train = sc.fit_transform(np.reshape(m_train, (m_train.shape[0], m_train.shape[1]*m_train.shape[2])))
    m_test = sc.fit_transform(np.reshape(m_test, (m_test.shape[0], m_test.shape[1]*m_test.shape[2])))
    print(f'loaded mnist: {m_train.shape} {m_test.shape}')

    # Load PIE data
    p_train = np.loadtxt(f'./data/PIE/StTrainFile{datasplit}.txt')
    p_test = np.loadtxt(f'./data/PIE/StTestFile{datasplit}.txt')
    p_train_y = p_train[:, -1] - 1
    p_train = sc.fit_transform(p_train[:, :-1])
    p_test_y = p_test[:, -1] - 1
    p_test = sc.fit_transform(p_test[:, :-1])
    print(f'loaded PIE: {p_train.shape} {p_test.shape}')

    # Load YaleB data
    yb_train = np.loadtxt(f'./data/YaleB/StTrainFile{datasplit}.txt')
    yb_test = np.loadtxt(f'./data/YaleB/StTestFile{datasplit}.txt')
    yb_train_y = yb_train[:, -1] - 1
    yb_train = sc.fit_transform(yb_train[:, :-1])
    yb_test_y = yb_test[:, -1] - 1
    yb_test = sc.fit_transform(yb_test[:, :-1])
    print(f'loaded YaleB: {yb_train.shape} {yb_test.shape}')
    
    metrics = {}

    print("\n\nWelcome to my classification program!")
    print("  [1] Support Vector Machines")
    print("  [2] XGBoost")
    print("  [3] Convolutional Neural Network")
    alg = input("Please enter the number of the algorithm you wish to use (q to quit): ")
    while alg != 'q':
        # SVM
        if alg == '1':
            print('\nrunning svm on YaleB...')
            yb_pred = run_svm(yb_train, yb_train_y, yb_test)
            metrics['YaleB_SVM'] = eval_performance(yb_test_y, yb_pred)
            
            print('\nrunning svm on PIE...')
            p_pred = run_svm(p_train, p_train_y, p_test)
            metrics['PIE_SVM'] = eval_performance(p_test_y, p_pred)

            print('\nrunning svm on MNIST...')
            p_pred = run_svm(p_train, p_train_y, p_test)
            metrics['PIE_SVM'] = eval_performance(p_test_y, p_pred)
        # XGBoost
        elif alg == '2':
            print('todo xgb')
        # CNN
        elif alg == '3':
            print('todo cnn')
        # Bad input
        else:
            alg = input("Only enter 1, 2, or 3 (q to quit): ")
            continue
        # Allow user to run code again
        alg = input("Please enter the number of the algorithm you wish to use (q to quit): ")
    print_metrics(metrics)

if __name__ == "__main__":
    main()
