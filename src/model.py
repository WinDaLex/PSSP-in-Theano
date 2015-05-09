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

    def __init__(self, n_input, n_hidden, n_output, L1_reg=0., L2_reg=0.0001):
        self.W_h = init_weights_sigmoid((n_input, n_hidden))
        self.b_h = init_bias(n_hidden)
        self.W_o = init_weights((n_hidden, n_output))
        self.b_o = init_bias(n_output)

        self.params = [self.W_h, self.b_h, self.W_o, self.b_o]
    
        self.X = T.matrix()
        self.Y = T.matrix()

        h = T.nnet.sigmoid(T.dot(self.X, self.W_h) + self.b_h)
        self.py_x =  T.nnet.softmax(T.dot(h, self.W_o) + self.b_o)

        self.NLL = -T.mean(T.log(self.py_x)[T.arange(self.Y.shape[0]), T.argmax(self.Y, axis=1)])
        self.L1 = T.sum(abs(self.W_h)) + T.sum(abs(self.W_o))
        self.L2_sqr = T.sum((self.W_h**2)) + T.sum((self.W_o**2))
        self.cost = self.NLL + L1_reg*self.L1 + L2_reg*self.L2_sqr

    def train(self, X, Y, learning_rate=0.001):
        grads = T.grad(cost=self.cost, wrt=self.params)
        updates = [[param, param - learning_rate*grad] for param, grad in zip(self.params, grads)]
        start = T.lscalar()
        end = T.lscalar()
        return theano.function(inputs=[start, end], outputs=self.cost, updates=updates, givens={ self.X: X[start:end], self.Y: Y[start:end]})

    def predict(self, X):
        start = T.lscalar()
        end = T.lscalar()
        return theano.function(inputs=[start, end], outputs=self.py_x, givens={self.X: X[start:end]})
