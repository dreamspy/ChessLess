# 3. Import libraries and modules
import numpy as np
np.random.seed(123)  # for reproducibility
 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import model_from_json

pieceMapW = {
  '1': 0,
  'p': 1,
  'r': 2,
  'n': 3,
  'b': 4,
  'q': 5,
  'k': 6,
  'P': 7,
  'R': 8,
  'N': 9,
  'B': 10,
  'Q': 11,
  'K': 12
}

pieceMapB = {
  '1': 0,
  'P': 1,
  'R': 2,
  'N': 3,
  'B': 4,
  'Q': 5,
  'K': 6,
  'p': 7,
  'r': 8,
  'n': 9,
  'b': 10,
  'q': 11,
  'k': 12  
}



def board64ToBoard768(nBoard):
    nB1 = np.copy(nBoard)
    nB2 = np.copy(nBoard)
    nB3 = np.copy(nBoard)
    nB4 = np.copy(nBoard)
    nB5 = np.copy(nBoard)
    nB6 = np.copy(nBoard)
    nB7 = np.copy(nBoard)
    nB8 = np.copy(nBoard)
    nB9 = np.copy(nBoard)
    nB10 = np.copy(nBoard)
    nB11 = np.copy(nBoard)
    nB12 = np.copy(nBoard)
    nB1[nB1 != 1] = 0 
    nB2[nB2 != 2] = 0 
    nB3[nB3 != 3] = 0 
    nB4[nB4 != 4] = 0 
    nB5[nB5 != 5] = 0 
    nB6[nB6 != 6] = 0 
    nB7[nB7 != 7] = 0 
    nB8[nB8 != 8] = 0 
    nB9[nB9 != 9] = 0 
    nB10[nB10 != 10] = 0 
    nB11[nB11 != 11] = 0 
    nB12[nB12 != 12] = 0 
    nBoard = np.hstack((nB1, nB2, nB3, nB4, nB5, nB6, nB7, nB8, nB9, nB10, nB11, nB12))
    return nBoard


def loadData(dataset,nRows):
    X_vals = []
    Y_vals = []
    i = 0
    pm = pieceMapW
    with open(dataset,'r') as data:
        boardLine = data.readline()

        while boardLine and i < nRows:
            boardData = boardLine.split(':')
            boardR = boardData[0].split('/')
            board = boardR[0:8]
            board[7] = board[7][0:8]
            who = boardR[7][9]
            
            if who == 'w':
                print()   
                nBoard = []
                nLine = []
                for line in board:
                    for piece in line:
                        nLine.append(pm[piece])
                    nBoard.append(nLine)
                    nLine = []
                nBoard = np.array(nBoard).reshape(64)
                # print(np.array(nBoard).reshape(64).shape)
                nBoard = board64ToBoard768(nBoard)
                X_vals.append(nBoard)
                #Y_vals.append(1 if float(boardData[2]) > 0 else 0)
                pc = float(boardData[1])
                Y_vals.append(pc)
            boardLine = data.readline()
            i += 1
        X_vals = np.array(X_vals)
        Y_vals = np.array(Y_vals)
#         print('wat',X_vals.shape)
#         print('Y', Y_vals[4:10])
        #Xraw = data.split('\n').map(lambda r: r.split('/'))

    # 4. Load pre-shuffled MNIST data into train and test sets

    # split = nRows // 2
#     (X_train, y_train) = X_vals[0:split], Y_vals[0: split].reshape(split,1)
#     (X_test, y_test) = X_vals[split :], Y_vals[split :].reshape(split,1)
    #(X_train, y_train), (X_test, y_test) = mnist.load_data()


    #n = 6000
    #X_train = X_train[0:n]
    #y_train = y_train[0:n]
    #X_test = X_test[0:n]
    #y_test = y_test[0:n]

    # 5. Preprocess input data
    # X_vals = X_vals.reshape(X_vals.shape[0], *chess_shape)
    
#     X_train = X_train.reshape(X_train.shape[0], *chess_shape)
#     X_test = X_test.reshape(X_test.shape[0], *chess_shape)
    
    #X_train = X_train.astype('float32')
    #X_test = X_test.astype('float32')
    #X_train /= 12
    # X_test /= 12


    print("done loading: ", dataset)
    return X_vals, Y_vals

chess_shape = (1, 8, 8)




X_test, Y_test = loadData('data/standar.data',2)
print(X_test)
print(X_test[0].shape)
print(Y_test)
print("done")



