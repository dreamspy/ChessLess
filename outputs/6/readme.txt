3 layers
32
1024
1024

all the data

 TESTING MODEL
 Classifier Error Rate:  0.442206849487045







model = Sequential()
model.add(Dense(32, activation='relu', input_dim=768))
model.add(Dropout(0.25, seed = 1337))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.25, seed = 1337))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.25, seed = 1337))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
                            metrics=['accuracy'])
                            # model.compile(loss='mean_squared_error', optimizer='adam')
                            model.summary()
