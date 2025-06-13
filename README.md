# StudentStudyHrsProject


**Note: this is my very first ANN or even a deep learning project, it is a very modest project based on a fake data that i asked chatgpt for**



*libraries used:*

-numpy

-pandas

-sklearn

-tensorflow



------
implementation code:




import numpy as np

import pandas as pd

import tensorflow as tf

dataset = pd.read_csv('simple_ann_dataset.csv')

X = dataset.iloc[:,0:2].values

y = dataset.iloc[:,-1].values     

print(X)

print(X)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)

ann = tf.keras.models.Sequential() #ANN creation

ann.add(tf.keras.layers.Dense(units = 6, activation = 'relu')) #input layer

ann.add(tf.keras.layers.Dense(units = 6, activation = 'relu')) #Neuron layer

ann.add(tf.keras.layers.Dense(units = 1, activation = 'sigmoid')) #output layer

ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])#ANN compiling

history=ann.fit(X_train, y_train, batch_size = 32, epochs = 100)# where the fun begins

y_pred = ann.predict(X_test)

y_pred = (y_pred > 0.5)

print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_test, y_pred)

print(cm)

accuracy_score(y_test, y_pred)

import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='accuracy')

plt.plot(history.history['loss'], label='loss')

plt.legend()

plt.show()

