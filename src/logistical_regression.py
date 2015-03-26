import theano
import theano.tensor as T
import numpy as np

import data


def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def init_weights(shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.01))

def model(X, w):
    return T.nnet.softmax(T.dot(X, w))

X_train, Y_train = data.load('data/training.data')
X_test, Y_test = data.load('data/test.data')

X = T.fmatrix()
Y = T.fmatrix()

w = init_weights((19*20, 3))

py_x = model(X, w)
y_pred = T.argmax(py_x, axis=1)

cost = T.mean(T.nnet.categorical_crossentropy(py_x, Y))
gradient = T.grad(cost=cost, wrt=w)
update = [[w, w - gradient * 0.05]]

train = theano.function(inputs=[X, Y], outputs=cost, updates=update, allow_input_downcast=True)
predict = theano.function(inputs=[X], outputs=y_pred, allow_input_downcast=True)

num_epoch = 10
for i in range(num_epoch):
    for start, end in zip(range(0, len(X_train), 8), range(8, len(X_train), 8)):
        cost = train(X_train[start:end], Y_train[start:end])
    print i, np.mean(np.argmax(Y_test, axis=1) == predict(X_test))
