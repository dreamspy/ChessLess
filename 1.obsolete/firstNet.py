# 3. Import libraries and modules
import numpy as np
np.random.seed(123)  # for reproducibility
 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist

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

X_vals = []
Y_vals = []

chess_shape = (1, 8, 8)

i, nRows = 0, 6000000
with open('standar.pgn.data','r') as data:
    boardLine = data.readline()
    
    while boardLine and i < nRows:
        boardData = boardLine.split(':')
        boardR = boardData[0].split('/')
        board = boardR[0:8]
        board[7] = board[7][0:8]
        
        who = boardR[7][9]
        
        nBoard = []
        nLine = []
        for line in board:
            for piece in line:
                pm = pieceMapW if who == 'w' else pieceMapB
                nLine.append(pm[piece])
            nBoard.append(nLine)
            nLine = []
        
        X_vals.append(nBoard)
        #Y_vals.append(1 if float(boardData[2]) > 0 else 0)
        Y_vals.append(float(boardData[2]))
        boardLine = data.readline()
        i += 1
    X_vals = np.array(X_vals)
    Y_vals = np.array(Y_vals)
    print('wat',X_vals.shape)
    print('Y', Y_vals[4:10])
    #Xraw = data.split('\n').map(lambda r: r.split('/'))
    #print(Xraw.shape)

# 4. Load pre-shuffled MNIST data into train and test sets

split = nRows // 2
(X_train, y_train) = X_vals[0:split], Y_vals[0: split].reshape(split,1)
(X_test, y_test) = X_vals[split :], Y_vals[split :].reshape(split,1)
#(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(X_train.shape)
#n = 6000
#X_train = X_train[0:n]
#y_train = y_train[0:n]
#X_test = X_test[0:n]
#y_test = y_test[0:n]

# 5. Preprocess input data
X_train = X_train.reshape(X_train.shape[0], *chess_shape)
X_test = X_test.reshape(X_test.shape[0], *chess_shape)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 12
X_test /= 12

#print(y_train)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
 
print("done")



# 6. Preprocess class labels
Y_train = np_utils.normalize(y_train)
Y_test = np_utils.normalize(y_test)

#Y_train = np_utils.to_categorical(y_train, 2)
#Y_test = np_utils.to_categorical(y_test, 2)
 
# 7. Define model architecture
model = Sequential()
 
# model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(1,28,28)))
#model.add(Dense(128, activation='relu', input_shape=chess_shape))
model.add(Convolution2D(16, (4, 4), activation='relu', input_shape=chess_shape, data_format='channels_first'))
model.add(Convolution2D(32, (2, 2), activation='relu'))
#model.add(MaxPooling2D(pool_size=(2,2)))
#model.add(Dropout(0.25))
 
model.add(Flatten())
model.add(Dense(1, activation='softsign'))
#model.compile(loss='categorical_crossentropy',
#              optimizer='adam',
#              metrics=['accuracy'])

print('compile')

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

print('summary')

model.summary()

# 9. Fit model on training data
print(X_train.shape, Y_train.shape)
model.fit(X_train, Y_train, 
          batch_size=32, epochs=10, verbose=1)

# 10. Evaluate model on test data
score = model.evaluate(X_test, y_test, verbose=0)
print("score: ",score)
