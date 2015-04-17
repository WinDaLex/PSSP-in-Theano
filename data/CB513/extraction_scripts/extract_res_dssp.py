import os
from os.path import join

def convert_dssp(ch):
    return {
        'G': 'H', 'I': 'H', 'H': 'H',
        'E': 'E',
        'T': 'C', 'B': 'C', 'S': 'C', '_': 'C', '?': 'C'
    }[ch];

path = 'raw/'

for files in os.listdir(path):
    with open(join(path, files), 'r') as f:
        print join(path, files)
        res_list = f.readline().strip()[4:].split(',')[:-1]
        dssp_list = f.readline().strip()[5:].split(',')[:-1]

        res_seq = ''.join(res_list)
        dssp_seq = ''.join(map(convert_dssp, dssp_list))

        # generate a data file
        print res_seq
        print dssp_seq
