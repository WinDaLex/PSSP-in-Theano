# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import theano
import theano.tensor as T

from measure import *

try:
    from itertools import izip
except ImportError:
    izip = zip

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


class MultilayerPerceptron(object):
    def __init__(self, n_input, n_hidden, n_output):
        print('... building model (%d-%d-%d)' %
                (n_input, n_hidden, n_output))

        self.W_h = init_weights_sigmoid((n_input, n_hidden))
        self.b_h = init_bias(n_hidden)
        self.W_o = init_weights((n_hidden, n_output))
        self.b_o = init_bias(n_output)

        self.params = [self.W_h, self.b_h, self.W_o, self.b_o]
    
        self.X = T.matrix()
        self.Y = T.matrix()

        h = T.nnet.sigmoid(T.dot(self.X, self.W_h) + self.b_h)
        self.py_x = T.nnet.softmax(T.dot(h, self.W_o) + self.b_o)

        y = T.argmax(self.Y, axis=1)
        self.NLL = -T.mean(T.log(self.py_x)[T.arange(self.Y.shape[0]), y])
        self.L1 = T.sum(abs(self.W_h)) + T.sum(abs(self.W_o))
        self.L2_sqr = T.sum((self.W_h**2)) + T.sum((self.W_o**2))
 
    def train_model(self, X_train, Y_train, X_valid, Y_valid,
                    num_epochs=3000, batch_size=20,
                    learning_rate=0.01, L1_reg=0., L2_reg=0.):

        print('... training model',
              '(batch_size: %d, learning_rate: %f, L1_reg: %f L2_reg: %f)' %
              (batch_size, learning_rate, L1_reg, L2_reg))

        cost = self.NLL + L1_reg*self.L1 + L2_reg*self.L2_sqr
 
        grads = T.grad(cost=cost, wrt=self.params)
        updates = [[param, param - learning_rate*grad]
                   for param, grad in izip(self.params, grads)]

        start = T.lscalar()
        end = T.lscalar()

        train = theano.function(
            inputs=[start, end],
            outputs=cost,
            updates=updates,
            givens={
                self.X: X_train[start:end],
                self.Y: Y_train[start:end]
            }
        )

        validate = theano.function(
            inputs=[start, end],
            outputs=[cost, self.py_x],
            givens={
                self.X: X_valid[start:end],
                self.Y: Y_valid[start:end]
            }
        )

        m_train = X_train.get_value(borrow=True).shape[0]
        m_valid = X_valid.get_value(borrow=True).shape[0]

        stopping_criteria = StoppingCriteria()
        index = range(0, m_train+1, batch_size)

        y_valid = np.argmax(Y_valid.get_value(borrow=True), axis=1)
        for i in range(num_epochs):
            costs = [train(index[j], index[j+1]) for j in range(len(index)-1)]
            E_tr = np.mean(costs)

            E_va, py_x = validate(0, m_valid)
            y_pred = np.argmax(py_x, axis=1)
            A_valid = AccuracyTable(y_pred, y_valid)

            stopping_criteria.append(E_tr, E_va)
            print('epoch %3d/%d. Cost: %f Q3_valid: %.2f%%' %
                    (i+1, num_epochs, E_tr, A_valid.Q3))

            if stopping_criteria.PQ(1):
                print('Early Stopping!')
                break

    def predict(self, X):
        start = T.lscalar()
        end = T.lscalar()
        return theano.function(
            inputs=[start, end],
            outputs=self.py_x,
            givens={self.X: X[start:end]}
        )
