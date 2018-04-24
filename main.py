import sys
import numpy as np
import tensorflow as tf

from model import dRNN
from reader1 import datastream
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

file = open('error.csv', 'w')

train_data, train_target, train_length = train_input.all(1.0)
test_data, test_target, test_length = test_input.all(1.0)

for step in range(steps):
    
    data, target, length = train_input.next(config().batch_size, step/steps)

    sess.run(m.optimize, feed_dict={x:data, y:target, l:length})

    if step % 100 == 0:
        train_err = sess.run(m.error,
                             {x:train_data, y:train_target, l:train_length})
        test_err = sess.run(m.error,
                             {x:test_data, y:test_target, l:test_length})
        file.write('{:3.10f}, {:3.10f}\n'.format(train_err, test_err))
        print('Epoch {:2d}'.format(step))

train_err = sess.run(m.error, {x:train_data, y:train_target, l:train_length})
test_err = sess.run(m.error, {x:test_data, y:test_target, l:test_length})

file.write('{:3.10f}, {:3.10f}'.format(train_err, test_err))
file.close()

print('Epoch {:2d} error {:3.1f}%'.format(step + 1, 100 * train_err))
print('Test set error {:3.1f}%'.format(100 * test_err))

