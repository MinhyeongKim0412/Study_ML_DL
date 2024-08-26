#%%
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

#############################################################
# %%
X = load_breast_cancer().data
Y = load_breast_cancer().target
# %%
X_train, X_test, Y_train, Y_test = train_test_split(X,Y)
# %%
X.shape
# %%
Y.shape
# %%
Y = Y.reshape(-1,1)
# %%
X_train.shape
# %%
model = Sequential()
model.add(Dense(256, activation = 'relu', input_shape=(30,)))
model.add(Dense(1, activation = 'sigmoid'))
# %%
model.summary()
# %%
model.predict(X_train).shape
# %%
model.compile(loss='binary_crossentropy', optimizer='SGD', metrics=['accuracy'])
# %%
model.fit(X_train, Y_train, epochs=10,
        validation_data=(X_test,Y_test),
        batch_size=64
        )
# %%
model.layers[0]
# %%
model.layers[0].get_weights()[0].shape
# %%
model.layers[1].get_weights()[0].shape
# %%
model.layers[0].get_weights()[1].shape
# %%
model.layers[1].get_weights()[1].shape
# %% ------------------------------------------------------------

def relu(X):
    return np.maximum(0,X)

def sigmoid(X):
    return 1/(1 + np.exp(-X))

class Network:
    
    def __init__(self):
        self.activation_dic = {
            'relu':relu,
            'sigmoid':sigmoid
        }
        self.layers = []
        self.activation = []
        
    def add(self,output,activation,input_shape=None):
        if input_shape == None:
            idx = len(self.layers)-1
            input_shape = self.layers[idx].shape[1]
            self.layers.append(np.random.randn(output))
        self.layers.append(np.random.randn(input_shape,output))
        self.activation.append(self.activation_dic[activation])

# ------------------------------------------------------------
# %%
model = Network()
model.add(256, activation='relu', input_shape=30)
model.add(100, activation='sigmoid')
# %%
model.layers[0].shape
# %%
model.activation[0](-1)
# %%
model = Sequential()
# %%
