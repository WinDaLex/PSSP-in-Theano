import re
import sys


source = sys.argv[1]
target = sys.argv[2]

with open(source, 'r') as source_file, open(target, 'w') as target_file:
    source_file.readline()

    for line in source_file:
        pattern = '"\d+","[^"]+","(\w+)","(\w+)"'
        prog = re.compile(pattern)
        result = prog.match(line)

        if result:
            primary = result.group(1)
            second_structure = result.group(2).replace('T', 'C')

            target_file.write(primary + '\n')
            target_file.write(second_structure + '\n')

            print primary
            print second_structure
        else:
            print 'Error: wrong format of the data file!'
