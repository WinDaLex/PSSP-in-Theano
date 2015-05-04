import numpy as np
import theano
import theano.tensor as T


def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def init_weights_sigmoid(shape):
    low = -np.sqrt(6./(shape[0]+shape[1])) * 4.
    high = np.sqrt(6./(shape[0]+shape[1])) * 4.
    values = np.random.uniform(low=low, high=high, size=shape)
    return theano.shared(floatX(values), borrow=True)

def init_weights(shape):
    values = np.random.randn(*shape)*0.01
    return theano.shared(floatX(values), borrow=True)

def init_bias(shape):
    values = np.zeros(shape, dtype=theano.config.floatX)
    return theano.shared(values, borrow=True)


class MultilayerPerceptron():

    def __init__(self, n_input, n_hidden, n_output, X):
        self.W_h = init_weights_sigmoid((n_input, n_hidden))
        self.b_h = init_bias(n_hidden)
        self.W_o = init_weights((n_hidden, n_output))
        self.b_o = init_bias(n_output)

        self.X = X

        self.py_x = self.model(X, self.W_h, self.b_h, self.W_o, self.b_o)
        self.y_pred = T.argmax(self.py_x, axis=1)

        self.params = [self.W_h, self.b_h, self.W_o, self.b_o]

    def model(self, X, W_h, b_h, W_o, b_o):
        h = T.nnet.sigmoid(T.dot(X, W_h) + b_h)
        return T.nnet.softmax(T.dot(h, W_o) + b_o)

    def negative_log_likelihood(self, Y):
        return -T.mean(T.log(self.py_x)[T.arange(Y.shape[0]), T.argmax(Y, axis=1)])

    @property
    def L1(self):
        return T.sum(abs(self.W_h)) + T.sum(abs(self.W_o))

    @property
    def L2_sqr(self):
        return T.sum((self.W_h**2)) + T.sum((self.W_o**2))
 
