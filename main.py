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
    print('\n')
    print(m)


def run_svm(train, train_y, test, ker='linear'):
    svm = SVC(kernel=ker) 
    svm.fit(train, train_y)
    return svm.predict(test)

def run_xgb(train, train_y, test, test_y):
    num_classes = len(np.unique(train_y))
    dtrain = xgb.DMatrix(train, label=train_y)
    dtest = xgb.DMatrix(test, label=test_y)
    param_list = { 
        "eta": 0.08, 
        "subsample": 0.8, 
        "colsample_bytree": 0.8, 
        "objective": "multi:softmax", 
        "eval_metric": "merror", 
        "alpha": 8, 
        "lambda": 2, 
        "num_class": num_classes
    }
    n_rounds = 100
    bst = xgb.train(param_list, dtrain, n_rounds)
    return bst.predict(dtest)

def run_cnn(train, train_y, test, kernel=3, ep=10):
    n = train.shape[1]
    num_classes = len(np.unique(train_y))
    cnn = Sequential()
    cnn.add(Conv2D(64, (kernel,kernel), input_shape=(n, n, 1), activation='relu'))
    cnn.add(MaxPooling2D(pool_size=(2,2)))
    cnn.add(Conv2D(64, (3,3), activation='relu'))
    cnn.add(MaxPooling2D(pool_size=(2,2)))
    cnn.add(Flatten())
    cnn.add(Dense(64, activation='relu'))
    cnn.add(Dense(num_classes, activation='softmax')) # depends on num classes
    cnn.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    train, val, train_y, val_y = train_test_split(train, train_y, test_size=0.2)
    cnn.fit(train, train_y, batch_size=64, epochs=ep, validation_data=(val, val_y), verbose=1)
    return cnn.predict_classes(test, batch_size=64)

#####################
# Main Program 
#####################
def main():
    ##########################
    # Load data
    ##########################
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
            print('\nrunning svm on YaleB...  ~3 sec')
            yb_pred = run_svm(yb_train, yb_train_y, yb_test)
            metrics['YaleB_SVM'] = eval_performance(yb_test_y, yb_pred)
            
            print('\nrunning svm on PIE...  ~30 sec')
            p_pred = run_svm(p_train, p_train_y, p_test)
            metrics['PIE_SVM'] = eval_performance(p_test_y, p_pred)

            print('\nrunning svm on MNIST... ~10 min')
            m_pred = run_svm(m_train, m_train_y, m_test, ker='rbf')
            metrics['MNIST_SVM'] = eval_performance(m_test_y, m_pred)
        
        # XGBoost
        elif alg == '2':
            print('\nrunning XGB on YaleB...  ~1 min')
            a=time.time()
            yb_pred = run_xgb(yb_train, yb_train_y, yb_test, yb_test_y)
            b=time.time()
            metrics['YaleB_XGB'] = eval_performance(yb_test_y, yb_pred)
            print(f'{(b-a)/60} min')

            print('\nrunning XGB on PIE...  ~9 min')
            a=time.time()
            p_pred = run_xgb(p_train, p_train_y, p_test, p_test_y)
            b=time.time()
            metrics['PIE_XGB'] = eval_performance(p_test_y, p_pred)
            print(f'{(b-a)/60} min')

            print('\nrunning XGB on MNIST...  ~7 min')
            a=time.time()
            m_pred = run_xgb(m_train, m_train_y, m_test, m_test_y)
            b=time.time()
            metrics['MNIST_XGB'] = eval_performance(m_test_y, m_pred)
            print(f'{(b-a)/60} min')

        # CNN
        elif alg == '3':

            print('\nrunning CNN on YaleB...  ~1 min')
            yb_pred = run_cnn(np.reshape(yb_train, (-1, 32, 32, 1)), yb_train_y, np.reshape(yb_test, (-1, 32, 32, 1)), kernel=5, ep=20)
            metrics['YaleB_CNN'] = eval_performance(yb_test_y, yb_pred)
            
            print('\nrunning CNN on PIE...  ~3 min')
            p_pred = run_cnn(np.reshape(p_train, (-1, 32, 32, 1)), p_train_y, np.reshape(p_test, (-1, 32, 32, 1)), kernel=5, ep=10)
            metrics['PIE_CNN'] = eval_performance(p_test_y, p_pred)
            
            print('\nrunning CNN on MNIST...  ~5 min')
            m_pred = run_cnn(np.reshape(m_train, (-1, 28, 28, 1)), m_train_y, np.reshape(m_test, (-1, 28, 28, 1)), ep=5)
            metrics['MNIST_CNN'] = eval_performance(m_test_y, m_pred)

        # Bad input
        else:
            alg = input("Only enter 1, 2, or 3 (q to quit): ")
            continue
        # Allow user to run code again
        alg = input("Please enter the number of the algorithm you wish to use (q to quit): ")
    
    print_metrics(metrics)

if __name__ == "__main__":
    main()
