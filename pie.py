import numpy as np

p_train = np.loadtxt('./data/PIE/StTrainFile1.txt')
p_test = np.loadtxt('./data/PIE/StTestFile1.txt')
# Handle reading in PIE data
for i in range(2,11):
    train_fp = f'./data/PIE/StTrainFile{i}.txt'
    test_fp = f'./data/PIE/StTestFile{i}.txt'
    train = np.loadtxt(train_fp)
    test = np.loadtxt(test_fp)
    print(f'tr: {p_train.shape}  tst: {p_test.shape}')
    np.concatenate((p_train, train))
    print(f'tr: {p_train.shape}  tst: {p_test.shape}')
    break