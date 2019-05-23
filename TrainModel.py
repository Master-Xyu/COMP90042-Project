from __future__ import absolute_import, division, print_function
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout
from keras.optimizers import Adadelta
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
    input.append([int(dic[key]['length'])/100, float(dic[key]['similarity1']), float(dic[key]['similarity2'])])
    output.append(float(dic[key]['label']))

train_input  = np.array(input,  dtype=float)
train_output = np.array(output,  dtype=float)

train_output=np_utils.to_categorical(train_output, 3)
t = 0
f = 0
n = 0
for asd in list(train_output):
    if asd[0] == 1:
        n += 1
    if asd[1] == 1:
        t += 1
    if asd[2] == 1:
        f += 1
print(t)
print(f)
print(n)
model=Sequential([
    Dense(input_dim=3,units=64),
    Activation('relu'),
    Dropout(0.1),
    Dense(units=128),
    Activation('relu'),
    Dropout(0.1),
    Dense(units=256),
    Activation('relu'),
    Dropout(0.1),
    Dense(3),
    Activation('softmax')
])

optimizer=Adadelta(lr=1.0, rho=0.95, epsilon=1e-06)

# 编译神经网络模型
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

# 训练模型
print('Training-------------------')

history =  model.fit(train_input,train_output,nb_epoch=200,batch_size=64)

print("Finished training the doc2vec_model")

model.save('vertification.h5')


