##### rms regressio##### rms regression


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
            
            pc = float(boardData[1])
            if (who == 'w') and (pc != 0) and (np.abs(pc) < 60):
                nBoard = []
                nLine = []
                for line in board:
                    for piece in line:
                        nLine.append(pm[piece])
                    nBoard.append(nLine)
                    nLine = []
                nBoard = np.array(nBoard).reshape(64)
                X_vals.append(board64ToBoard768(nBoard))
                Y_vals.append(pc)
            boardLine = data.readline()
            i += 1
        X_vals = np.array(X_vals)
        Y_vals = np.array(Y_vals)
    print("done loading: ", dataset)
    return X_vals, Y_vals

def measureErrorRates(full):
    print("\n TESTING MODEL")
    if full:
        # n = 10000
        # X_predict, Y_predict = loadData('data/standar.data',64*n)
        X_predict = X_test
        Y_predict = Y_test
    else:
        X_predict = X_test[0:100]
        Y_predict = Y_test[0:100]

    cnnPredict = model.predict(X_predict)
    error = Y_predict-cnnPredict[0]
    errorSum = 0
    for i in range(len(error)):
        errorSum += abs(error[i])
    Y_predict = (np.array(Y_predict))
    cnnPredict = (np.array(cnnPredict.reshape(len(Y_predict))))
    error = (np.array(error.reshape(len(Y_predict))))

    # print("INPUT - OUTPUT - ERROR")
    # print(np.vstack((Y_predict,cnnPredict,error)).T)
    # print("SUM ABS(ERROR) = ",errorSum)

    errorCount = 0
    for i in range(len(Y_predict)):
    	if abs(error[i]) > 0.5 : errorCount += 1
    print("Value Error Rate (with respect to +- 0.5 difference): ", errorCount/len(Y_predict) )

    signError = 0
    for i in range(len(Y_predict)):
        if cnnPredict[i] * Y_predict[i] < 0: signError +=1
    print("Sign Error Rate: ", signError/len(Y_predict))



chess_shape = (1, 8, 8)

# # # 7. Define model architecture
dim = 512
model = Sequential()
model.add(Dense(units=dim, input_dim=768, use_bias = True))
model.add(Activation('relu'))
model.add(Dropout(0.25, seed = 1337))
model.add(Dense(units=dim, use_bias = True))
model.add(Activation('relu'))
model.add(Dropout(0.25, seed = 1337))
model.add(Dense(units=dim, use_bias = True))
model.add(Activation('relu'))
model.add(Dense(1, activation='linear'))

model.compile(loss='mean_squared_error', optimizer='adam')
model.summary()


# X_train, Y_train = loadData('data/smallSample.data',20)
# model.fit(X_train, Y_train, batch_size=32, epochs=10, verbose=1)

# LOAD TEST DATA
#m=1: small sample
#m=1000: all data
m = 10000
print("Loading testdata  standar")
X_test, Y_test = loadData('data/standar.data',64*m)

# BLITZZ 
print("\nLoading blitzz")
X_train, Y_train = loadData('data/blitzz.data',130*m)
print("\nFitting blitzz")
model.fit(X_train, Y_train, batch_size=32, epochs=10, verbose=1)

print("evaluating")
score = model.evaluate(X_test, Y_test, verbose=1)
print("score: ",score)
measureErrorRates(False)

# LIGHTNING 
print("\nLoading lightning")
X_train, Y_train = loadData('data/lightning.data',89*m)
print("\nFitting lightning")
model.fit(X_train, Y_train, batch_size=32, epochs=10, verbose=1)

print("evaluating")
score = model.evaluate(X_test, Y_test, verbose=1)
print("score: ",score)
measureErrorRates(False)



# TITLED 
print("\nLoading titled")
X_train, Y_train = loadData('data/titled.data',320*m)
print("\nFitting titled")
model.fit(X_train, Y_train, batch_size=32, epochs=10, verbose=1)

print("evaluating")
score = model.evaluate(X_test, Y_test, verbose=1)
print("score: ",score)
measureErrorRates(False)



# SAVING DATA
# serialize model to JSON
model_json = model.to_json()
with open("model_1.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model_1.h5")
print("Saved model to disk")

# LOADING DATA
# load json and create model
# json_file = open('model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)
# # load weights into new model
# loaded_model.load_weights("model.h5")
# print("Loaded model from disk")

#do full error rates measure
measureErrorRates(True)
