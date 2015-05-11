import cPickle
import numpy as np
import theano
import theano.tensor as T

import data
from model import MultilayerPerceptron
from measure import AccuracyTable
from measure import StoppingCriteria


def transform(x, m, window_size=17):
    double_end = [0.] * 3 * (window_size / 2)
    sequences = double_end + x.tolist()[0] + double_end
    return [sequences[index:index+window_size*3] for index in xrange(0, m*3, 3)]


network_file = '2015-05-10 22:46:33.nn'
train_file = 'data/astral30.pssm'
valid_file = 'data/casp9.pssm'

window_size = 17

with open(network_file, 'rb') as f:
    fst_layer_classifier = cPickle.load(f)

def get_XY(filename):
    X_data, Y, index = data.shared_dataset(data.load_pssm(filename))
    m = X_data.get_value(borrow=True).shape[0]
    predict = fst_layer_classifier.predict(X_data)
    x = predict(0, m).reshape(1, m*3)
    x = transform(x, m, window_size)
    X = theano.shared(data.floatX(x), borrow=True)
    return X, Y, index

X_train, Y_train, index_train = get_XY(train_file)
X_valid, Y_valid, index_valid = get_XY(valid_file)

snd_layer_classifier = MultilayerPerceptron(window_size*3, window_size*3, 3)
train = snd_layer_classifier.train(X_train, Y_train, learning_rate=1)
valid = snd_layer_classifier.predict2(X_valid, Y_valid)

batch_size = 20
num_epochs = 1000

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

    print 'epoch %3d\%d. train cost: %f, Q3_valid: %.3f%%. %f %f %f' % \
        (i+1, num_epochs, np.mean(costs), A_valid.Q3, E_va, stopping_criteria.generalization_loss, stopping_criteria.training_progress)

    if stopping_criteria.UP(2):
        print 'Stop Early!'
        break

