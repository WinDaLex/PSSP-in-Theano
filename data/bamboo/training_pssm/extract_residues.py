import sys

filename = sys.argv[1]

with open(filename, 'r') as f:
    i = 0
    for line in f:
        if i % 2 == 0:
            with open("%d.res" % (i/2), 'w') as target:
                target.write(line)
        i += 1

