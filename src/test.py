import cPickle
import numpy as np
import theano
import theano.tensor as T

import data
from measure import AccuracyTable

test_file = 'data/casp9.pssm'

X_test, Y_test, index_test = data.shared_dataset(data.load_pssm(test_file))
m_test = len(index_test) - 1

with open('2015-05-10 22:46:33.nn', 'rb') as f:
    classifier = cPickle.load(f)

predict = classifier.predict(X_test)
py_x = predict(0, m_test)
Y_pred = np.argmax(py_x, axis=1)
Y_obs = np.argmax(Y_test.get_value(borrow=True), axis=1)
A_test = AccuracyTable(Y_pred, Y_obs)

print A_test.Q3
