## @package translate_dataset
#  Translates the encoded data into a readable json format for data verification
#  Usage: python3 read_dataset.py \<inputfile\>

import pickle
import json
import sys

if len(sys.argv) > 1:
    ## The file to read
    input = sys.argv[1]
else:
    print("Usage: python3 read_dataset.py <inputfile>")
    sys.exit()

## The input file object
fin = open(input, 'rb')
## The output file object
fout = open('{}.json'.format(input), 'w')

while True:
    try:
        json.dump(pickle.load(fin), fout, sort_keys=True, indent=4)
        fout.write('\n')
    except EOFError:
        break
        
fin.close()
fout.close()
