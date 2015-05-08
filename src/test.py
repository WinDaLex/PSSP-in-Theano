import cPickle
import numpy as np
import theano
import theano.tensor as T

import data
from measure import AccuracyTable

test_file = 'data/casp9.pssm'

X_test, Y_test, index_test = data.load_pssm(test_file)

with open('obj.save', 'rb') as f:
    classifier = cPickle.load(f)

predict = theano.function(
    inputs=[classifier.X],
    outputs=classifier.y_pred,
    allow_input_downcast=True
)

Y_pred = predict(X_test)
Y_obs = np.argmax(Y_test, axis=1)
A_test = AccuracyTable(Y_pred, Y_obs)

print A_test.Q3
