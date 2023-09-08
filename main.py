import tensorflow
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense

model = Sequential([
    Dense(units=25, activation='relu'),
    Dense(units=15, activation='relu'),
    Dense(units=10, activation='linear')
])

from tensorflow.python.keras.losses import SparseCategoricalCrossentropy

model.compile(loss=SparseCategoricalCrossentropy(from_logits=True))

# model.fit(X, Y, )
