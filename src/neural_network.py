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
        outputs=[cost, classifier.y_pred],
        givens={
            X: X_valid[start:end],
            Y: Y_valid[start:end]
        },
        allow_input_downcast=True
    )

    return train, predict


def train_model(num_epochs=1, batch_size=1):
    print '... training model (batch_size = %d)' % batch_size
 
    m_train = X_train.get_value(borrow=True).shape[0]
    m_valid = X_valid.get_value(borrow=True).shape[0]


    stopping_threshold = 3
    validation_frequency = 5
    best_validation_loss = np.inf
    stopping_count = 0

    losses = []
    index = range(0, m_train+1, batch_size)
    for i in range(num_epochs):
        A_train = AccuracyTable()
        for j in range(len(index) - 1):
            loss, Y_pred = train(index[j], index[j+1])
            losses.append(loss)
            Y_obs = np.argmax(Y_train.get_value(borrow=True)[index[j]:index[j+1]], axis=1)
            A_train.count(Y_pred, Y_obs)

        print 'epoch %3d. Loss: %f, Q3_train: %.3f%%.' % \
            (i+1, np.average(losses), A_train.Q3)

        if ((i + 1) % validation_frequency == 0):
            this_validation_loss, Y_pred = predict(0, m_valid)
            Y_obs = np.argmax(Y_valid.get_value(borrow=True), axis=1)
            A_valid = AccuracyTable(Y_pred, Y_obs)
            print '%f %.3f%%' % (this_validation_loss, A_valid.Q3)
            if this_validation_loss < best_validation_loss:
                best_validation_loss = this_validation_loss
                stopping_count = 0
            else:
                stopping_count += 1

        if stopping_count >= stopping_threshold:
            break


def shared_dataset(data_xy, borrow=True):
    data_x, data_y, index = data_xy
    shared_x = theano.shared(floatX(data_x), borrow=borrow)
    shared_y = theano.shared(floatX(data_y), borrow=borrow)
    return shared_x, shared_y, index

if __name__ == '__main__':
    print datetime.datetime.now()
    if len(sys.argv) >= 2:
        print "Label:", sys.argv[1]

    train_file = 'data/astral30.pssm'
    valid_file = 'data/casp9.pssm'
    window_size = 19

    hidden_layer_size = 100
    learning_rate = 0.03

    num_epochs = 5
    batch_size = 20

    X_train, Y_train, index_train = shared_dataset(data.load_pssm(train_file, window_size=window_size))
    X_valid, Y_valid, index_valid = shared_dataset(data.load_pssm(valid_file, window_size=window_size))

    train, predict = build_model(window_size=window_size, hidden_layer_size=hidden_layer_size, learning_rate=learning_rate)

    train_model(num_epochs=num_epochs, batch_size=batch_size)

    with open(str(datetime.datetime.now())[:19] + '.nn', 'wb') as f:
        cPickle.dump(classifier, f)

    print '\nDone!'
