import sys

import json

data = json.load(open(sys.argv[1]))
o = open(sys.argv[2], 'w')

for prob in data:
    print prob['sQuestion'], prob['lSolutions']
    o.write(prob['sQuestion'])
    o.write(' ')
    o.write(str(prob['lSolutions'][0]))
    o.write("\n")
    #o.write(prob['sQuestion'] + ' ' + str(prob['lSolutions'][0]) + "\n")

o.close()

