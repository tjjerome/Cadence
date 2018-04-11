import pickle
import json
import sys

fin = open('data', 'rb')
fout = open('data.json', 'w')

while True:
    try:
        json.dump(pickle.load(fin), fout, sort_keys=True, indent=4)
        fout.write('\n')
    except EOFError:
        break
        
fin.close()
fout.close()
