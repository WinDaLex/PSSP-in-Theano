import sys

for i in range(1, len(sys.argv)):
    source_file = sys.argv[i]
    target_file = source_file.split('/')[1].split('.')[0] + '.data'

    with open(source_file, 'r') as source, open(target_file, 'w') as target:
        source.readline()
        source.readline()
        source.readline()

        for line in source:
            if len(line) >= 69:
                target.write(line[9:69] + '\n')
