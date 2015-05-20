#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import argparse
import datetime

import numpy as np
import theano
import theano.tensor as T

try:
    import configparser
    import pickle
except ImportError:
    import ConfigParser as configparser
    import cPickle as pickle
    from itertools import izip as zip
    input = raw_input
    range = xrange


def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def piecewise_scaling_func(x):
    if x < -5:
        y = 0.0
    elif -5 <= x <= 5:
        y = 0.5 + 0.1*x
    else:
        y = 1.0
    return y


class DataLoader(object):

    @staticmethod
    def encode_residue(residue):
        return [1 if residue == amino_acid else 0
                for amino_acid in ('A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H',
                                   'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W',
                                   'Y', 'V')]

    @staticmethod
    def encode_dssp(dssp):
        return [1 if dssp == hec else 0 for hec in ('H', 'E', 'C')]

    @staticmethod
    def shared_dataset(data_xy, borrow=True):
        data_x, data_y, index = data_xy
        shared_x = theano.shared(floatX(data_x), borrow=borrow)
        shared_y = theano.shared(floatX(data_y), borrow=borrow)
        return shared_x, shared_y, index

    @staticmethod
    def load(filename, window_size=19):
        print('... loading data ("%s")' % filename)

        X = []
        Y = []
        index = [0]
        with open(filename, 'r') as f:
            line = f.read().strip().split('\n')
            num_proteins = len(line) // 2

            for line_num in range(num_proteins):
                sequence = line[line_num*2]
                structure = line[line_num*2 + 1]

                double_end = [None] * (window_size // 2)
                unary_sequence = []
                for residue in double_end + list(sequence) + double_end:
                    unary_sequence += DataLoader.encode_residue(residue)

                X += [
                    unary_sequence[start: start+window_size*20]
                    for start in range(0, len(sequence)*20, 20)
                ]

                Y += [DataLoader.encode_dssp(dssp) for dssp in structure]

                index.append(index[-1] + len(sequence))

        return DataLoader.shared_dataset([X, Y, index])

    @staticmethod
    def load_pssm(filename, window_size=19, scale=piecewise_scaling_func):
        print('... loading pssm ("%s")' % filename)

        X = []
        Y = []
        index = [0]
        with open(filename, 'r') as f:
            num_proteins = int(f.readline().strip())
            for __ in range(num_proteins):
                m = int(f.readline().strip())
                sequences = []
                for __ in range(m):
                    line = f.readline()
                    sequences += [scale(float(line[i*3: i*3+3]))
                                  for i in range(20)]

                double_end = ([0.]*20) * (window_size//2)
                sequences = double_end + sequences + double_end
                X += [
                    sequences[start:start+window_size*20]
                    for start in range(0, m*20, 20)
                ]

                structure = f.readline().strip()
                Y += [DataLoader.encode_dssp(dssp) for dssp in structure]

                index.append(index[-1] + m)

        return DataLoader.shared_dataset([X, Y, index])


class AccuracyTable(object):

    def __init__(self, pred=None, obs=None):
        self.table = np.zeros(shape=(3, 3), dtype=float)
        if pred is not None and obs is not None:
            self.count(pred, obs)

    def count(self, pred, obs):
        for i in range(len(pred)):
            self.table[obs[i]][pred[i]] += 1

    @property
    def Q3(self):
        return self.table.trace() / self.table.sum() * 100

    def correlation_coefficient(self, p, n, o, u):
        return (p*n-o*u) / ((p+o)*(p+u)*(n+o)*(n+u))**0.5

    @property
    def Ch(self):
        p = self.table[0][0]
        n = self.table[1][1] + self.table[2][2]
        o = self.table[1][0] + self.table[2][0]
        u = (self.table[0][1] + self.table[0][2] + self.table[1][2] +
             self.table[2][1])
        return self.correlation_coefficient(p, n, o, u)

    @property
    def Ce(self):
        p = self.table[1][1]
        n = self.table[0][0] + self.table[2][2]
        o = self.table[0][1] + self.table[2][1]
        u = (self.table[1][0] + self.table[2][0] + self.table[0][2] +
             self.table[1][2])
        return self.correlation_coefficient(p, n, o, u)

    @property
    def Cc(self):
        p = self.table[2][2]
        n = self.table[0][0] + self.table[1][1]
        o = self.table[0][2] + self.table[1][2]
        u = (self.table[1][0] + self.table[2][0] + self.table[0][1] +
             self.table[2][1])
        return self.correlation_coefficient(p, n, o, u)

    @property
    def C3(self):
        return np.mean((self.Ch, self.Ce, self.Cc))


class StoppingCriteria(object):
    def __init__(self, k=5):
        self.t = 0
        self.k = k
        self.E_tr = [np.inf]
        self.E_va = [np.inf]
        self.E_opt = np.inf

    def append(self, E_tr, E_va):
        self.t += 1
        self.E_tr.append(E_tr)
        self.E_va.append(E_va)
        self.E_opt = min(self.E_opt, E_va)

    @property
    def generalization_loss(self):
        return 100. * (self.E_va[-1]/self.E_opt - 1)

    @property
    def training_progress(self):
        return 1000. * (sum(self.E_tr[-self.k:]) /
                        (self.k * min(self.E_tr[-self.k:])) - 1)

    def GL(self, alpha):
        """Stop as soon as the generalization loss exceeds a certain threshold.
        """
        return self.generalization_loss > alpha

    def PQ(self, alpha):
        """Stop as soon as quotient of generalization loss and progress exceeds
        a certain threshold
        """
        return self.generalization_loss / self.training_progress > alpha

    def UP(self, s, t=0):
        """Stop when the generalization error increased in s successive strips.
        """
        if t == 0:
            t = self.t
        if t - self.k < 0 or self.E_va[t] <= self.E_va[t - self.k]:
            return False
        if s == 1:
            return True
        return self.UP(s - 1, t - self.k)


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
        print('... building model (%d-%d-%d)' % (n_input, n_hidden, n_output))

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
                   for param, grad in zip(self.params, grads)]

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


def first_tier():
    current_time = datetime.datetime.now()
    print(current_time)

    config = configparser.RawConfigParser()
    config.read('first-level.cfg')

    train_file = config.get('FILE', 'training_file')
    valid_file = config.get('FILE', 'validation_file')

    window_size = config.getint('MODEL', 'window_size')
    hidden_layer_size = config.getint('MODEL', 'hidden_layer_size')

    learning_rate = config.getfloat('TRAINING', 'learning_rate')
    L1_reg = config.getfloat('TRAINING', 'l1_reg')
    L2_reg = config.getfloat('TRAINING', 'l2_reg')
    num_epochs = config.getint('TRAINING', 'num_epochs')
    batch_size = config.getint('TRAINING', 'batch_size')

    X_train, Y_train, index_train = DataLoader.load_pssm(train_file,
                                                         window_size=window_size)
    X_valid, Y_valid, index_valid = DataLoader.load_pssm(valid_file,
                                                         window_size=window_size)

    input_layer_size = window_size * 20
    output_layer_size = 3

    classifier = MultilayerPerceptron(input_layer_size,
                                      hidden_layer_size,
                                      output_layer_size)

    classifier.train_model(X_train, Y_train, X_valid, Y_valid,
                           num_epochs, batch_size,
                           learning_rate, L1_reg, L2_reg)

    network_file = str(current_time)[5:16] + '.nn'
    print('... saving model in file (%s)' % network_file)
    pickle.dump(classifier, open(network_file, 'wb'))

    print('Done!')


def second_tier():

    def transform(x, m, window_size=17):
        double_end = [0.] * 3 * (window_size // 2)
        sequences = double_end + x.tolist()[0] + double_end
        return [sequences[index: index+window_size*3]
                for index in range(0, m*3, 3)]

    def get_XY(filename):
        X_data, Y_data, index = DataLoader.load_pssm(filename)
        m = X_data.get_value(borrow=True).shape[0]
        predict = fst_layer_classifier.predict(X_data)
        x = predict(0, m).reshape(1, m*3)
        x = transform(x, m, window_size)
        X = theano.shared(floatX(x), borrow=True)
        return X, Y_data, index

    current_time = datetime.datetime.now()
    print(current_time)

    config = configparser.RawConfigParser()
    config.read('second-level.cfg')

    network_file = config.get('FILE', 'network_file')
    train_file = config.get('FILE', 'training_file')
    valid_file = config.get('FILE', 'validation_file')

    window_size = config.getint('MODEL', 'window_size')
    hidden_layer_size = config.getint('MODEL', 'hidden_layer_size')

    learning_rate = config.getfloat('TRAINING', 'learning_rate')
    L1_reg = config.getfloat('TRAINING', 'l1_reg')
    L2_reg = config.getfloat('TRAINING', 'l2_reg')

    num_epochs = config.getint('TRAINING', 'num_epochs')
    batch_size = config.getint('TRAINING', 'batch_size')

    fst_layer_classifier = pickle.load(open(network_file, 'rb'))

    X_train, Y_train, index_train = get_XY(train_file)
    X_valid, Y_valid, index_valid = get_XY(valid_file)

    snd_layer_classifier = MultilayerPerceptron(window_size*3,
                                                hidden_layer_size,
                                                3)
    snd_layer_classifier.train_model(X_train, Y_train, X_valid, Y_valid,
                                     num_epochs, batch_size,
                                     learning_rate, L1_reg, L2_reg)

    network_file = str(current_time)[5:16] + '.nn'
    print('... saving model in file (%s)' % network_file)
    pickle.dump(snd_layer_classifier, open(network_file, 'wb'))

    print('Done!')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-q', '--quiet', action='store_true', help='stay quiet, do not print anything')
    parser.add_argument('-V', '--version', action='store_true', help='show version.')
    parser.add_argument('--verbose', '-v', action='store_true')
    subparsers = parser.add_subparsers(dest='command')

    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('level', choices=['first', 'second', 'all'])

    parser_test = subparsers.add_parser('test')
    parser_predict = subparsers.add_parser('predict')

    args = parser.parse_args()
    if args.command == 'train':
        if args.level == 'first' or 'all':
            first_tier()
        if args.level == 'second' or 'all':
            second_tier()

if __name__ == '__main__':
    main()
