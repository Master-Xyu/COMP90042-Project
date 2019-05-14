from __future__ import absolute_import, division, print_function
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.optimizers import RMSprop
from keras.utils import np_utils
import json

tf.logging.set_verbosity(tf.logging.ERROR)

import numpy as np

file = open('train_output.txt', 'r')
js = file.read()
dic = json.loads(js)

input = []
output = []
for key in dic.keys():
    input.append([int(dic[key]['length'])/100, float(dic[key]['similarity'])])
    output.append(float(dic[key]['label']))

train_input    = np.array(input,  dtype=float)
train_output = np.array(output,  dtype=float)

train_output=np_utils.to_categorical(train_output, 3)

model=Sequential([
    Dense(input_dim=2,units=128),
    Activation('relu'),
    Dense(units=64),
    Activation('relu'),
    Dense(units=32),
    Activation('relu'),
    Dense(3),
    Activation('softmax')
])

rmsprop=RMSprop(lr=0.001,rho=0.9,epsilon=1e-08,decay=0.0)

# 编译神经网络模型
model.compile(loss='mean_squared_error',
              optimizer=rmsprop,
              metrics=['accuracy'])

# 训练模型
print('Training-------------------')
model.fit(train_input,train_output,nb_epoch=100,batch_size=32)
print("Finished training the model")

model.save('vertification.h5')

print(model.predict(np.array([[0.1, 0.9666593478936601]],  dtype=float)))

