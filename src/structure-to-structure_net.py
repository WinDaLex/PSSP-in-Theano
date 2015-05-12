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

import theano

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
snd_layer_classifier.train_model(X_train, Y_train, X_valid, Y_valid, num_epochs, learning_rate, batch_size)
