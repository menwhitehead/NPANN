import sys

f = open(sys.argv[1], 'r')
o = open(sys.argv[2], 'w')
for line in f:
  for c in line:
    c = c.lower()
    if c.isalpha() or c == ' ':
      o.write(c)
o.close()
f.close()
