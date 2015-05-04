import sys
import datetime
import cPickle
import theano
import theano.tensor as T
import numpy as np

import data
from measure import AccuracyTable
from model import MultilayerPerceptron

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def build_model(window_size=19, hidden_layer_size=100, learning_rate=0.03,
                L1_reg=0.00, L2_reg=0.0001):

    global classifier

    def sgd(cost, params):
        grads = T.grad(cost=cost, wrt=params)
        updates = []
        for param, grad in zip(params, grads):
            updates.append([param, param - learning_rate*grad])
        return updates

    print '... building model (%d-%d-%d)' % (window_size*20, hidden_layer_size, 3)

    X = T.matrix()
    Y = T.matrix()

    input_layer_size = window_size * 20
    output_layer_size = 3

    classifier = MultilayerPerceptron(input_layer_size, hidden_layer_size, output_layer_size, X)
    cost = classifier.negative_log_likelihood(Y) + L1_reg*classifier.L1 + L2_reg*classifier.L2_sqr

    updates = sgd(cost, classifier.params)

    start = T.lscalar()
    end = T.lscalar()

    train = theano.function(
        inputs=[start, end],
        outputs=[cost, classifier.y_pred],
        updates=updates,
        givens={
            X: X_train[start:end],
            Y: Y_train[start:end]
        },
        allow_input_downcast=True
    )

    predict = theano.function(
        inputs=[start, end],
        outputs=classifier.y_pred,
        givens={
            X: X_test[start:end]
        },
        allow_input_downcast=True
    )

    return train, predict


def train_model(num_epochs=1, batch_size=1):
    print '... training model (batch_size = %d)' % batch_size
 
    m_train = X_train.get_value(borrow=True).shape[0]
    m_test = X_test.get_value(borrow=True).shape[0]

    index = range(0, m_train+1, batch_size)

    cost_list = []
    for i in range(num_epochs):
        A_train = AccuracyTable()
        for j in range(len(index) - 1):
            cost, Y_pred = train(index[j], index[j+1])
            cost_list.append(cost)
            Y_obs = np.argmax(Y_train.get_value(borrow=True)[index[j]:index[j+1]], axis=1)
            A_train.count(Y_pred, Y_obs)

        Y_pred = predict(0, m_test)
        Y_obs = np.argmax(Y_test.get_value(borrow=True), axis=1)
        A_test = AccuracyTable(Y_pred, Y_obs)

        print 'epoch %3d/%d. Loss: %f, Q3_train: %.3f%%, Q3_test: %.3f%%.' % \
            (i+1, num_epochs, np.average(cost_list), A_train.Q3, A_test.Q3)

def shared_dataset(data_xy, borrow=True):
    data_x, data_y, index = data_xy
    shared_x = theano.shared(floatX(data_x), borrow=borrow)
    shared_y = theano.shared(floatX(data_y), borrow=borrow)
    return shared_x, shared_y, index

if __name__ == '__main__':
    print datetime.datetime.now()
    if len(sys.argv) >= 2:
        print "Label:", sys.argv[1]

    train_file = 'data/casp9_pssm.data'
    test_file = 'data/casp9_pssm.data'
    window_size = 19

    hidden_layer_size = 100
    learning_rate = 0.03

    num_epochs = 10
    batch_size = 20

    X_train, Y_train, index_train = shared_dataset(data.load_pssm(train_file, window_size=window_size))
    X_test, Y_test, index_test = shared_dataset(data.load_pssm(test_file, window_size=window_size))

    train, predict = build_model(window_size=window_size, hidden_layer_size=hidden_layer_size, learning_rate=learning_rate)

    train_model(num_epochs=num_epochs, batch_size=batch_size)

    with open('obj.save', 'wb') as f:
        cPickle.dump(classifier, f)

    print '\nDone!'
