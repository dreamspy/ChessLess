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
                nBoard = []
                nLine = []
                for line in board:
                    for piece in line:
                        nLine.append(pm[piece])
                    nBoard.append(nLine)
                    nLine = []
                nBoard = np.array(nBoard).reshape(64)
                # print(np.array(nBoard).reshape(64).shape)
                X_vals.append(nBoard)
                #Y_vals.append(1 if float(boardData[2]) > 0 else 0)
                pc = float(boardData[1])
                Y_vals.append(pc)
#                 print(boardLine)
#                 print(nBoard)
#                 print(pc,"\n")
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




# X_test, Y_test = loadData('data/standar.data',4)
# print(X_test[0].shape)
# print(Y_test)




# # 7. Define model architecture
model = Sequential()
model.add(Dense(units=64, input_dim=64, use_bias = True))
model.add(Activation('relu'))
model.add(Dropout(0.25, seed = 1337))
model.add(Dense(units=64, input_dim=64, use_bias = True))
model.add(Activation('relu'))
model.add(Dropout(0.25, seed = 1337))
model.add(Dense(units=64, input_dim=64, use_bias = True))
model.add(Activation('relu'))
# model.add(Dense(units=64, input_dim=64, use_bias = True))
# model.add(Activation('relu'))
# model.add(Dropout(0.25, seed = 1337))
# model.add(Dense(units=64, input_dim=64, use_bias = True))
# model.add(Activation('relu'))
model.add(Dense(1, activation='linear'))

print('compile')
model.compile(loss='mean_squared_error', optimizer='adam')
print('summary')
model.summary()


# X_train, Y_train = loadData('data/smallSample.data',20)
# model.fit(X_train, Y_train, batch_size=32, epochs=10, verbose=1)

#m=1: small sample
# #m=1000: all data
m= 10000
# LOAD TEST DATA
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


# LIGHTNING 
print("\nLoading lightning")
X_train, Y_train = loadData('data/lightning.data',89*m)
print("\nFitting lightning")
model.fit(X_train, Y_train, batch_size=32, epochs=10, verbose=1)

print("evaluating")
score = model.evaluate(X_test, Y_test, verbose=1)
print("score: ",score)



# TITLED 
print("\nLoading titled")
X_train, Y_train = loadData('data/titled.data',320*m)
print("\nFitting titled")
model.fit(X_train, Y_train, batch_size=32, epochs=10, verbose=1)

print("evaluating")
score = model.evaluate(X_test, Y_test, verbose=1)
print("score: ",score)


print("Done\n")


def testSmallSample():
  print("\n TESTING MODEL")
  X_predict, Y_predict = loadData('data/standar.data',100)
  cnnPredict = model.predict(X_predict)
  error = Y_predict-cnnPredict[0]
  errorSum = 0
  for i in range(len(error)):
    errorSum += abs(error[i])
  Y_predict = (np.array(Y_predict))
  cnnPredict = (np.array(cnnPredict.reshape(len(Y_predict))))
  error = (np.array(error.reshape(len(Y_predict))))
  print("INPUT - OUTPUT - ERROR")
  print(np.vstack((Y_predict,cnnPredict,error)).T)
  print("SUM ABS(ERROR) = ",errorSum)
  errorCount = 0
  for i in range(len(Y_predict)):
    if abs(error[i]) > 0.5 : errorCount += 1
  print("Error Rate: ", errorCount/len(Y_predict) )
testSmallSample()
# SAVING DATA
# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

# load json and create model
# json_file = open('model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)
# # load weights into new model
# loaded_model.load_weights("model.h5")
# print("Loaded model from disk")
