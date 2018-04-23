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

mean, std = train_input.norm_params()
train_input.normalize(mean, std)
test_input.normalize(mean, std)

x = tf.placeholder("float", [None, max_length, train_input.num_features])
y = tf.placeholder(tf.int32, [None, max_length])
l = tf.placeholder(tf.int32, [None])

m = dRNN(x, y, l, config())

init = tf.global_variables_initializer()

sess = tf.Session()

sess.run(init)

for step in range(steps):
    
    data, target, length = train_input.next(config().batch_size)

    sess.run(m.optimize, feed_dict={x:data, y:target, l:length})

    if step % 200 == 0:
        print('Epoch {:2d}'.format(step))

data, target, length = train_input.all()
incorrect = sess.run(m.error, {x:data, y:target, l:length})
print('Epoch {:2d} error {:3.1f}%'.format(step + 1, 100 * incorrect))

data, target, length = test_input.all()
incorrect = sess.run(m.error, {x:data, y:target, l:length})
print('Test set error {:3.1f}%'.format(100 * incorrect))
