# -*- coding: utf-8 -*-

from __future__ import print_function

import sys
import datetime

try:
    import ConfigParser as configparser
    import cPickle as pickle
except ImportError:
    import configparser
    import pickle

import data
from model import MultilayerPerceptron


print(datetime.datetime.now())
if len(sys.argv) >= 2:
    print("Label:", sys.argv[1])

# load config file

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

# load dataset

X_train, Y_train, index_train = data.load_pssm(train_file, window_size=window_size)
X_valid, Y_valid, index_valid = data.load_pssm(valid_file, window_size=window_size)

# build model

print('... building model (%d-%d-%d)' % (window_size*20, hidden_layer_size, 3))

input_layer_size = window_size * 20
output_layer_size = 3

classifier = MultilayerPerceptron(input_layer_size, hidden_layer_size, output_layer_size, L1_reg, L2_reg)

# train model

classifier.train_model(X_train, Y_train, X_valid, Y_valid, num_epochs, learning_rate, batch_size)

# save model

filename = str(datetime.datetime.now())[:19] + '.nn'
print('... saving model in file (%s)' % filename)

pickle.dump(classifier, open(filename, 'wb'))

print('\nDone!')
