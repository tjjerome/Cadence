import sys
import numpy as np
import tensorflow as tf

from model import dRNN
from reader import datastream
from config import config

##############################################################################

# Run Session

steps = config().training_steps
max_length = config().max_length

train_input = datastream('train', config())

x = tf.placeholder("float", [None, max_length, train_input.num_features])
y = tf.placeholder("float", [None, max_length, train_input.num_features])
length = tf.placeholder(tf.int32, [None])

m = dRNN(x, y, config())

init = tf.global_variables_initializer()

sess = tf.Session()

sess.run(init)
album, albumlen = train_input.next()

for step in range(steps):

    sess.run(m.optimize, feed_dict={x:album, y:album, length:albumlen})

    print(sess.run(m.error, feed_dict={x:album, y:album, length:albumlen}))
