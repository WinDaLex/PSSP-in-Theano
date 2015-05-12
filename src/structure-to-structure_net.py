from __future__ import print_function

try:
    import ConfigParser as configparser
    import cPickle as pickle
except ImportError:
    import configparser
    import pickle

try:
    input = raw_input
    range = xrange
except NameError:
    pass

import numpy as np
import theano
import theano.tensor as T

import data
from model import MultilayerPerceptron
from measure import AccuracyTable
from measure import StoppingCriteria


def get_XY(filename):
    X_data, Y, index = data.load_pssm(filename)
    m = X_data.get_value(borrow=True).shape[0]
    predict = fst_layer_classifier.predict(X_data)
    x = predict(0, m).reshape(1, m*3)
    x = transform(x, m, window_size)
    X = theano.shared(data.floatX(x), borrow=True)
    return X, Y, index

def transform(x, m, window_size=17):
    double_end = [0.] * 3 * (window_size / 2)
    sequences = double_end + x.tolist()[0] + double_end
    return [sequences[index : index+window_size*3] for index in range(0, m*3, 3)]

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

fst_layer_classifier = pickle.load(open(network_file, 'r'))

X_train, Y_train, index_train = get_XY(train_file)
X_valid, Y_valid, index_valid = get_XY(valid_file)

snd_layer_classifier = MultilayerPerceptron(window_size*3, hidden_layer_size, 3)
train = snd_layer_classifier.train(X_train, Y_train, learning_rate=learning_rate)
valid = snd_layer_classifier.validate(X_valid, Y_valid)

m_train = X_train.get_value(borrow=True).shape[0]
m_valid = X_valid.get_value(borrow=True).shape[0]

stopping_criteria = StoppingCriteria()

index = range(0, m_train+1, batch_size)
for i in range(num_epochs):
    costs = [train(index[j], index[j+1]) for j in range(len(index)-1)]
    E_tr = np.mean(costs)

    E_va, py_x = valid(0, m_valid)
    y_pred = np.argmax(py_x, axis=1)
    y_obs = np.argmax(Y_valid.get_value(borrow=True)[0:m_valid], axis=1)
    A_valid = AccuracyTable(y_pred, y_obs)

    stopping_criteria.append(E_tr, E_va)

    print('epoch %3d\%d. train cost: %f, Q3_valid: %.3f%%. %f %f %f' % \
        (i+1, num_epochs, np.mean(costs), A_valid.Q3, E_va, stopping_criteria.generalization_loss, stopping_criteria.training_progress))

    if stopping_criteria.PQ(1):
        print('Stop Early!')
        break

