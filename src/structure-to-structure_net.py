import cPickle
import numpy as np
import theano
import theano.tensor as T

import data
from model import MultilayerPerceptron
from measure import AccuracyTable

network_file = 'obj.save'

with open(network_file, 'rb') as f:
    fst_layer_classifier = cPickle.load(f)

train_file = 'data/astral30.pssm'

X_train, Y_train, index_train = data.shared_dataset(data.load_pssm(train_file))

start = T.lscalar()
end = T.lscalar()

predict = theano.function(
    inputs=[start, end],
    outputs=fst_layer_classifier.py_x,
    givens={
        fst_layer_classifier.X: X_train[start:end]
    },
    allow_input_downcast=True
)

m = X_train.get_value(borrow=True).shape[0]

x = predict(0, m).reshape(1, m*3)

window_size = 17
double_end = [0., 0., 0.] * (window_size / 2)

sequences = double_end + x.tolist()[0] + double_end

x_train = [sequences[begin:begin+window_size*3] for begin in xrange(0,m*3,3)]

#print x_train

X_train = theano.shared(data.floatX(x_train), borrow=True)

# train model

learning_rate = 0.001

def sgd(cost, params):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for param, grad in zip(params, grads):
        updates.append([param, param - learning_rate*grad])
    return updates

X = T.matrix()
Y = T.matrix()

snd_layer_classifier = MultilayerPerceptron(17*3, 17*3, 3, X)

cost = snd_layer_classifier.negative_log_likelihood(Y)
updates = sgd(cost, snd_layer_classifier.params)

start = T.lscalar()
end = T.lscalar()

train = theano.function(
    inputs=[start, end],
    outputs=snd_layer_classifier.y_pred,
    updates=updates,
    givens={
        X: X_train[start:end],
        Y: Y_train[start:end]
    },
    allow_input_downcast=True
)

X_test, Y_test, index_test = data.shared_dataset(data.load_pssm('data/casp9.pssm'))

predict = theano.function(
    inputs=[start, end],
    outputs=fst_layer_classifier.py_x,
    givens={
        fst_layer_classifier.X: X_test[start:end]
    },
    allow_input_downcast=True
)

m_test = X_test.get_value(borrow=True).shape[0]

x = predict(0, m_test).reshape(1, m_test*3)

window_size = 17
double_end = [0., 0., 0.] * (window_size / 2)

sequences = double_end + x.tolist()[0] + double_end

x_test = [sequences[begin:begin+window_size*3] for begin in xrange(0,m_test*3,3)]

#print x_train

start = T.lscalar()
end = T.lscalar()

X_test = theano.shared(data.floatX(x_test), borrow=True)


test = theano.function(
    inputs=[start, end],
    outputs=snd_layer_classifier.y_pred,
    givens={
        X: X_test[start:end]
    },
    allow_input_downcast=True
)



# test

batch_size = 20
num_epochs = 1000

m_test = X_test.get_value(borrow=True).shape[0]

index = range(0, m+1, batch_size)
for i in range(num_epochs):
    A_train = AccuracyTable()
    for j in range(len(index) - 1):
        Y_pred = train(index[j], index[j+1])
        Y_obs = np.argmax(Y_train.get_value(borrow=True)[index[j]:index[j+1]], axis=1)
        A_train.count(Y_pred, Y_obs)

    Y_pred = test(0, m_test)
    Y_obs = np.argmax(Y_test.get_value(borrow=True)[0:m_test], axis=1)
    A_test = AccuracyTable(Y_pred, Y_obs)

    print 'epoch %3d\%d. Q3_train: %.3f%%, Q3_test: %.3f%%.' % \
        (i+1, num_epochs, A_train.Q3, A_test.Q3)
