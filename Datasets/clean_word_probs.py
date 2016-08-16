
import string

f = open("word_probs.txt", 'r')
o = open("cleaned.txt", 'w')

for line in f:
    tokens = line.split()
    answer = tokens[-1]
    txt = tokens[:-1]
    for toke in txt:
        toke = toke.lower()
        toke = toke.strip(".")
        toke = toke.replace(",", '')
        toke = toke.replace(";", '')
        toke = toke.strip("'")
        toke = toke.strip("$")
        toke = toke.strip("?")
        toke = toke.strip()
        if toke[-2:] == "'s":
            toke = toke[:-2]
        if len(toke) == 1 and toke[0] == 's':
            continue
        if len(toke) > 0:
            o.write(toke + " ")

    o.write(answer + "\n")

    #new_txt = ''
    #for c in txt:
    #    c = c.lower()
    #    if c.isalpha() or c.isdigit() or c == ' ':
    #        new_txt += c
    #    else:
    #        new_txt += ' '

    #tokens = new_txt.split()
    #o.write(string.join(tokens, ' ') + ' ' + answer + "\n")

f.close()
o.close()
