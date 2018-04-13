import sys
import pickle
import numpy as np
import tensorflow as tf

def str2int(s):
    chars = "0123456789qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM"
    i = 0
    for c in reversed(s):
        i *= len(chars)
        i += chars.index(c)
    return i % 1000003

def get_song_vector(song):
    song['id'] = str2int(song['id'])
    vec = list(song.items())
    vec.sort()
    for i in range(len(vec)):
        vec[i] = float(vec[i][1])
    return vec

###############################################################################

# Defining data

class datastream():
    def __init__(self, inputfile):
        fin = open(inputfile, 'rb')
        self.max_length = 49
        self.batch_size = 10
        album_stream = []
        while True:
            try:
                album = pickle.load(fin)
                song_stream = []
                
                for i in range(self.max_length):
                    try:
                        song_stream.append(get_song_vector(album[i]))
                    except IndexError:
                        song_stream.append([0] * len(song_stream[0]))
                        
                album_stream.append(song_stream)
            except EOFError:
                break
            
        self.num_features = len(song_stream[0])
        
        album_stream = np.array(album_stream)
        
        all_data = tf.data.Dataset.from_tensor_slices(album_stream)
        dataset = all_data.batch(self.batch_size)
        self.iterator = dataset.make_initializable_iterator()
        self.next_batch = tf.transpose(self.iterator.get_next(), [1,0,2])

# Building a model



"""lstm = tf.contrib.rnn.BasicLSTMCell(max_length)
hidden_state = tf.zeros([None, lstm.state_size])
current_state = tf.zeros([None, lstm.state_size])
state = hidden_state, current_state"""

probabilities = []
loss = 0.0

# Loss


# Define Training


# Run Session
sess = tf.Session()

train_input = datastream('train')

sess.run(train_input.iterator.initializer)
print(sess.run(train_input.next_batch).shape)
