import matplotlib
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras

from keras import Sequential
from keras.layers import Dense, Conv1D
from keras.regularizers import L2
from keras.losses import MeanSquaredError

matplotlib.rcParams['figure.figsize'] = [9, 6]

x = tf.linspace(-2, 2, 201)
x = tf.cast(x, tf.float32)


def f(x):
    y = x ** 2 + 2 * x - 5
    return y


y = f(x) + tf.random.normal(shape=[201])

plt.plot(x.numpy(), y.numpy(), '.', label='Data')
plt.plot(x, f(x), label='Ground truth')

model = Sequential([
    Dense(units=25, activation='relu'),
    Dense(units=20, activation='relu'),
    Dense(units=1, activation='linear')
])

x = tf.convert_to_tensor(x)
y = tf.convert_to_tensor(y)

print(x.shape)
print(y.shape)
model.compile(loss=MeanSquaredError(), optimizer='adam')

model.fit(tf.expand_dims(x, axis=-1), y, epochs=5000)

predictions = model.predict(x)

plt.plot(x, predictions, label='prediction')
plt.legend()
plt.show()

