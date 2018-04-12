import sys
import pickle
import numpy as np
import tensorflow as tf

max_length = 49
batch_size = 10

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

fin = open('train', 'rb')

album_stream = []

while True:
    try:
        album = pickle.load(fin)
        song_stream = []
    
        for i in range(max_length):
            try:
                song_stream.append(get_song_vector(album[i]))
            except IndexError:
                song_stream.append([0] * len(song_stream[0]))
        
        album_stream.append(song_stream)
    except EOFError:
        break
    
length = len(song_stream)
num_features = len(song_stream[0])

album_stream = np.array(album_stream)

#album_stream = np.swapaxes(album_stream,0,1)
print(album_stream.shape)

weights = {
    'out': tf.Variable(tf.random_normal([max_length]))
}
biases = {
    'out': tf.constant(0.0)
}

songs = tf.placeholder("float")

all_data = tf.data.Dataset.from_tensor_slices(songs)
dataset = all_data.batch(batch_size)
iterator = dataset.make_initializable_iterator()
next_song = iterator.get_next()

# Building a graph

"""lstm = tf.contrib.rnn.BasicLSTMCell(max_length)
hidden_state = tf.zeros([None, lstm.state_size])
current_state = tf.zeros([None, lstm.state_size])
state = hidden_state, current_state"""

probabilities = []
loss = 0.0

# Loss


# Run Session
sess = tf.Session()

sess.run(iterator.initializer, feed_dict={songs: album_stream})
print(sess.run(next_song).shape)
