import pickle
import sys
import numpy as np

if len(sys.argv) > 1:
    input = sys.argv[1]
else:
    print("Usage: python3 split_dataset.py <inputfile>")
    sys.exit()

fin = open(input, 'rb')
test = open('test', 'wb')
train = open('train', 'wb')

n = 0
train_n = 0
test_n = 0

while True:
    try:
        if np.random.binomial(1,0.2) == 0:
            pickle.dump(pickle.load(fin), train)
            train_n += 1
            
        else:
            pickle.dump(pickle.load(fin), test)
            test_n += 1
            
        n += 1
        
    except EOFError:
        break

print('{} albums processed'.format(n))
print('Train - {}, Test - {}'.format(train_n, test_n))
        
fin.close()
test.close()
train.close()
