import sys
import datetime
import cPickle
import ConfigParser

import numpy as np
import theano
import theano.tensor as T

import data
from measure import AccuracyTable
from measure import StoppingCriteria
from model import MultilayerPerceptron


print datetime.datetime.now()
if len(sys.argv) >= 2:
    print "Label:", sys.argv[1]

# load config file

config = ConfigParser.RawConfigParser()
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

print '... building model (%d-%d-%d)' % (window_size*20, hidden_layer_size, 3)

input_layer_size = window_size * 20
output_layer_size = 3

classifier = MultilayerPerceptron(input_layer_size, hidden_layer_size, output_layer_size, L1_reg, L2_reg)
train = classifier.train(X_train, Y_train, learning_rate)
predict = classifier.validate(X_valid, Y_valid)

# train model

print '... training model (batch_size = %d, learning_rate = %f)' % (batch_size, learning_rate)

m_train = X_train.get_value(borrow=True).shape[0]
m_valid = X_valid.get_value(borrow=True).shape[0]

stopping_criteria = StoppingCriteria()

index = range(0, m_train+1, batch_size)
for i in range(num_epochs):
    costs = [train(index[j], index[j+1]) for j in range(len(index)-1)]
    E_tr = np.mean(costs)

    E_va, py_x = predict(0, m_valid)
    y_pred = np.argmax(py_x, axis=1)
    y_obs = np.argmax(Y_valid.get_value(borrow=True), axis=1)
    A_valid = AccuracyTable(y_pred, y_obs)

    stopping_criteria.append(E_tr, E_va)
    print 'epoch %3d/%d. Cost: %f Q3_valid: %.2f%%' % (
        i+1, num_epochs,
        E_tr, A_valid.Q3),
    print 'E_valid: %.4f, GL: %.4f, TR: %.4f' % (
        stopping_criteria.E_va[-1],
        stopping_criteria.generalization_loss,
        stopping_criteria.training_progress)

    if stopping_criteria.PQ(1):
        print 'Early Stopping!'
        break

# save model

filename = str(datetime.datetime.now())[:19] + '.nn'
print '... saving model in file (%s)' % filename
with open(filename, 'wb') as f:
    cPickle.dump(classifier, f)

print '\nDone!'
