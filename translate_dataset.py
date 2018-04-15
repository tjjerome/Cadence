import pickle
import json
import sys

if len(sys.argv) > 1:
    input = sys.argv[1]
else:
    print("Usage: python3 read_dataset.py <inputfile>")
    sys.exit()

fin = open(input, 'rb')
fout = open('{}.json'.format(input), 'w')

while True:
    try:
        json.dump(pickle.load(fin), fout, sort_keys=True, indent=4)
        fout.write('\n')
    except EOFError:
        break
        
fin.close()
fout.close()
