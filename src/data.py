# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import theano

try:
    input = raw_input
    range = xrange
except NameError:
    pass

def encode_residue(residue):
    return [1 if residue == amino_acid else 0
            for amino_acid in ('A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H',
                               'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W',
                               'Y', 'V')]

def encode_dssp(dssp):
    return [1 if dssp == hec else 0 for hec in ('H', 'E', 'C')]

def load(filename, window_size=19):
    print('... loading data ("%s")' % filename)

    X = []
    Y = []
    index = [0]
    with open(filename, 'r') as f:
        line = f.read().strip().split('\n')
        num_proteins = len(line) // 2

        for line_num in range(num_proteins):
            sequence = line[line_num*2]
            structure = line[line_num*2 + 1]

            double_end = [None] * (window_size // 2)
            unary_sequence = []
            for residue in double_end + list(sequence) + double_end:
                unary_sequence += encode_residue(residue)

            X += [
                unary_sequence[start : start+window_size*20]
                for start in range(0, len(sequence)*20, 20)
            ]

            Y += [encode_dssp(dssp) for dssp in structure]

            index.append(index[-1] + len(sequence))

    data_x = X
    data_y = Y
    shared_x = theano.shared(floatX(data_x), borrow=True)
    shared_y = theano.shared(floatX(data_y), borrow=True)
    return shared_x, shared_y, index

def piecewise_scaling_func(x):
    if x < -5:
        y = 0.0
    elif -5 <= x <= 5:
        y = 0.5 + 0.1*x
    else:
        y = 1.0
    return y

def load_pssm(filename, window_size=19, scale=piecewise_scaling_func):
    print('... loading pssm ("%s")' % filename)

    X = []
    Y = []
    index = [0]
    with open(filename, 'r') as f:
        num_proteins = int(f.readline().strip())
        for __ in range(num_proteins):
            m = int(f.readline().strip())
            sequences = []
            for __ in range(m):
                line = f.readline()
                sequences += [scale(float(line[i*3 : i*3+3])) for i in range(20)]

            double_end = ([0.]*20) * (window_size//2)
            sequences = double_end + sequences + double_end
            X += [
                sequences[start:start+window_size*20]
                for start in range(0, m*20, 20)
            ]

            structure = f.readline().strip()
            Y += [encode_dssp(dssp) for dssp in structure]

            index.append(index[-1] + m)

    data_x = X
    data_y = Y
    shared_x = theano.shared(floatX(data_x), borrow=True)
    shared_y = theano.shared(floatX(data_y), borrow=True)
    return shared_x, shared_y, index

def floatX(A):
    return np.asarray(A, dtype=theano.config.floatX)

def shared_dataset(data_xy, borrow=True):
    data_x, data_y, index = data_xy
    shared_x = theano.shared(floatX(data_x), borrow=borrow)
    shared_y = theano.shared(floatX(data_y), borrow=borrow)
    return shared_x, shared_y
