from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import os
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten,LSTM,TimeDistributed
from keras.layers import Convolution2D, MaxPooling2D,MaxPooling1D,Conv1D
from keras.optimizers import Adam,SGD
from keras.utils import np_utils
from sklearn import metrics
import random
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import optimizers
import datetime


def reSample(data, samples):
    r = len(data)/samples #re-sampling ratio
    newdata = []
    for i in range(0,samples):
        newdata.append(data[int(i*r)])
    return np.array(newdata)
  

train_subjects = ['s01']
test_subjects = ['s02']

def get_data(path,sampleSize):
    # 2 types of event present in audio
    activities = ['tring', 'calling']
    
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []
    
    #STFT features located in "STFT_features/stft_257_1/" directory   
    for file in os.listdir(path + 'stft_257_1/'):
        print(len(file.split("_")[1].split("_")[0]))
        if int(file.split("_")[1].split("_")[0])!=1:
            a = (np.load(path + "stft_257_1/" + file)).T
            label = file.split('_')[-1].split(".")[0]
            print("label=" +str(label))
            if file.split("_")[0] in train_subjects:
                X_train.append(np.mean(a,axis=0))
                Y_train.append(label)
            else:
                X_test.append(np.mean(a,axis=0))
                Y_test.append(label)          
                  
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)
    
    print(X_train)
    print(Y_train)
    print(X_test)
    print(Y_test)
    
    return X_train,Y_train,X_test,Y_test
  
def print_M(conf_M):
    s = "activity,"
    for i in range(len(conf_M)):
        s += lb.inverse_transform([i])[0] + ","
    print(s[:-1])
    for i in range(len(conf_M)):
        s = ""
        for j in range(len(conf_M)):
            s += str(conf_M[i][j])
            s += ","
        print(lb.inverse_transform([i])[0],",", s[:-1])
    print()
        
def print_M_P(conf_M):
    s = "activity,"
    for i in range(len(conf_M)):
        s += lb.inverse_transform([i])[0] + ","
    print(s[:-1])
    for i in range(len(conf_M)):
        s = ""
        for j in range(len(conf_M)):
            val = conf_M[i][j]/float(sum(conf_M[i]))
            s += str(round(val,2))
            s += ","
        print(lb.inverse_transform([i])[0],",", s[:-1])
    print()        
        
def showResult():
    predictions = [np.argmax(y) for y in result]
    expected = [np.argmax(y) for y in y_test]

    conf_M = []
    num_labels=y_test[0].shape[0]
    for i in range(num_labels):
        r = []
        for j in range(num_labels):
            r.append(0)
        conf_M.append(r)

  

    n_tests = len(predictions)
    for i in range(n_tests):
        conf_M[expected[i]][predictions[i]] += 1

    print_M(conf_M)
    print_M_P(conf_M)


featuresPath = "STFT_features/"

a,b,c,d = get_data(featuresPath,250)


X_train,Y_train,X_test,Y_test = a,b,c,d

n_samples = len(Y_train)
print("No of training samples: " + str(n_samples))
order = np.array(range(n_samples))
np.random.shuffle(order)
X_train = X_train[order]
Y_train = Y_train[order]

lb = LabelEncoder()
y_train = np_utils.to_categorical(lb.fit_transform(Y_train))
y_test = np_utils.to_categorical(lb.fit_transform(Y_test))

num_labels = y_train.shape[1]


num_labels = y_train.shape[1]
filter_size = 2
y_train = y_train.astype('int')


# build model
model = Sequential()

model.add(Dense(256, input_shape=(257,)))
model.add(Activation('relu'))
# model.add(Dropout(0.5))

model.add(Dense(256))
model.add(Activation('relu'))
# model.add(Dropout(0.5))

model.add(Dense(128))
model.add(Activation('relu'))
# model.add(Dropout(0.5))

model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(num_labels))
model.add(Activation('softmax'))

model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer='adam')
# model.summary()

model.fit(X_train, y_train, batch_size=10, epochs=60)

result = model.predict(X_test)
#print(result)


## save model (optional)
path = "Models/audio_NN_New"+datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
model_json = model.to_json()
with open(path+"_acc_"+acc+".json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights(path+"_acc_"+acc+".h5")




