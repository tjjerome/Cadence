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
test_input = datastream('test', config())

x = tf.placeholder("float", [None, max_length, train_input.num_features])
y = tf.placeholder("float", [None, max_length, max_length])
l = tf.placeholder(tf.int32, [None])

m = dRNN(x, y, l, config())

init = tf.global_variables_initializer()

sess = tf.Session()

sess.run(init)

for step in range(steps):
    data, target, length = train_input.next(config().batch_size)

    sess.run(m.optimize, feed_dict={x:data, y:target, l:length})

    if step % 200 == 0:
        loss = sess.run(m.error, {x:data, y:target, l:length})
        print('Epoch {:2d} loss {:3.1f}'.format(step, loss))

print(sess.run(m.prediction, feed_dict={x:data, y:target, l:length}))
#incorrect = sess.run(m.error, {x:data, y:target})
#print('Epoch {:2d} error {:3.1f}%'.format(step + 1, 100 * incorrect))
