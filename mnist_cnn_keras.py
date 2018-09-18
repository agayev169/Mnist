import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0

x_train = tf.reshape(x_train, shape = [-1, 28, 28, 1])
x_test = tf.reshape(x_test, shape = [-1, 28, 28, 1])

model = Sequential()

model.add(Conv2D(32, (2, 2), input_shape = (28, 28, 1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(32, (2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.1))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.1))

model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(optimizer = 'adam', 
	loss = 'sparse_categorical_crossentropy',
	metrics = ['accuracy'])

model.fit(x_train, y_train, epochs = 20, batch_size = 128, validation_data = (x_test, y_test))
# model.evaluate(x_test, y_test)
