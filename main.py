import keras.src.regularizers
import tensorflow as tf
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense, Conv1D
from tensorflow.python.keras.regularizers import L2

model = Sequential([
    Dense(units=25, activation='relu', kernel_regularizer=L2(0.02)),
    Dense(units=15, activation='relu', kernel_regularizer=L2(0.02)),
    Dense(units=10, activation='linear', kernel_regularizer=L2(0.02))
])


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))

# model.fit(X, Y, )
