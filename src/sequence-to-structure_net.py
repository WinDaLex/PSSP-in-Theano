import sys
import datetime
import cPickle
import theano
import theano.tensor as T
import numpy as np

import data
from data import floatX
from measure import AccuracyTable
from measure import StoppingCriteria
from model import MultilayerPerceptron

def train_model(num_epochs=1, batch_size=1):
    print '... training model (batch_size = %d, learning_rate = %f)' % (batch_size, learning_rate)
 
    m_train = X_train.get_value(borrow=True).shape[0]
    m_valid = X_valid.get_value(borrow=True).shape[0]

    stopping_criteria = StoppingCriteria()

    index = range(0, m_train+1, batch_size)
    for i in range(num_epochs):
        costs = [train(index[j], index[j+1]) for j in range(len(index)-1)]
        E_tr = np.mean(costs)
        print 'epoch %3d. Cost: %f' % (i+1, np.mean(costs))

        E_va, py_x = predict(0, m_valid)
        y_pred = np.argmax(py_x, axis=1)
        y_obs = np.argmax(Y_valid.get_value(borrow=True), axis=1)
        A_valid = AccuracyTable(y_pred, y_obs)
 
        stopping_criteria.append(E_tr, E_va)
        print stopping_criteria.E_va[-1], stopping_criteria.generalization_loss, stopping_criteria.training_progress

        if stopping_criteria.GL(1):
            break

print datetime.datetime.now()
if len(sys.argv) >= 2:
    print "Label:", sys.argv[1]

train_file = 'data/astral30.pssm'
valid_file = 'data/casp9.pssm'
window_size = 19
hidden_layer_size = 100

learning_rate = 10000
L1_reg = 0.
L2_reg = 0.0000

num_epochs = 1000
batch_size = 20

X_train, Y_train, index_train = data.shared_dataset(data.load_pssm(train_file, window_size=window_size))
X_valid, Y_valid, index_valid = data.shared_dataset(data.load_pssm(valid_file, window_size=window_size))

print '... building model (%d-%d-%d)' % (window_size*20, hidden_layer_size, 3)

input_layer_size = window_size * 20
output_layer_size = 3

classifier = MultilayerPerceptron(input_layer_size, hidden_layer_size, output_layer_size, L1_reg, L2_reg)
train = classifier.train(X_train, Y_train)
predict = classifier.predict2(X_valid, Y_valid)

train_model(num_epochs=num_epochs, batch_size=batch_size)

with open(str(datetime.datetime.now())[:19] + '.nn', 'wb') as f:
    cPickle.dump(classifier, f)

print '\nDone!'
