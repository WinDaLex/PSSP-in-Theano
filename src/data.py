RESIDUES_CLASS = ('A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I', 'L', 'K',
                  'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V')

DSSP_CLASS = ('H', 'E', 'C')


def encode_residue(residue):
    return [1 if residue == RESIDUES_CLASS[i] else 0 for i in xrange(20)]


def encode_dssp(dssp):
    return [1 if dssp == DSSP_CLASS[i] else 0 for i in xrange(3)]


def load(filename, window_size=19):
    print '... loading data ("%s")' % filename

    X = []
    Y = []
    index = [0]
    with open(filename, 'r') as f:
        line = f.read().strip().split('\n')
        num_proteins = len(line) / 2

        for line_num in xrange(num_proteins):
            sequence = line[line_num*2]
            structure = line[line_num*2 + 1]

            double_end = [None] * (window_size / 2)
            unary_sequence = []
            for residue in double_end + list(sequence) + double_end:
                unary_sequence += encode_residue(residue)

            X += [
                unary_sequence[start:start+window_size*20]
                for start in xrange(0, len(sequence)*20, 20)
            ]

            Y += [encode_dssp(dssp) for dssp in structure]

            index.append(index[-1] + len(sequence))

    return X, Y, index
