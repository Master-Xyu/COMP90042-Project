from __future__ import absolute_import, division, print_function
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

import numpy as np

# 输入的摄氏温度以及其对应的华氏温度，前面两个故意写错了
celsius_q    = np.array([-40, -10,  0,  8, 15, 22,  38],  dtype=float)
fahrenheit_a = np.array([-40,  14, 32, 46, 59, 72, 100],  dtype=float)

for i,c in enumerate(celsius_q):
    print("{} degrees Celsius = {} degrees Fahrenheit".format(c, fahrenheit_a[i]))

# 输入后连接了只有一个神经单元的隐藏层
l0 = tf.keras.layers.Dense(units=1, input_shape=[1])

# 将网络层添加到序列模型中
model = tf.keras.Sequential([l0])

# 编译神经网络模型
model.compile(loss='mean_squared_error',
              optimizer=tf.keras.optimizers.Adam(0.1))

# 训练模型
history = model.fit(celsius_q, fahrenheit_a, epochs=500, verbose=False)
print("Finished training the model")

print(model.predict([100.0]))