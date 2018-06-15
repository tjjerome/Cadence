## @package main
#  Execution of the tensorflow session
#  @author Trevor Jerome

import sys
import numpy as np
import tensorflow as tf

from model import dRNN
from reader import datastream
from config import config

##############################################################################

# Run Session

## Stores the number of training steps to be performed, defined in config.py
steps = config().training_steps
## Stores the max length of the album, defined in config.py
max_length = config().max_length

## A datastream object for training data
train_input = datastream('train', config())
## A datastream object for testing data
test_input = datastream('test', config())

## @{Vectors of means and standard deviations for each song feature
mean, std = train_input.norm_params()
## @}
train_input.normalize(mean, std)
test_input.normalize(mean, std)

## Tensorflow placeholder for input data
x = tf.placeholder("float", [None, max_length, train_input.num_features])
## Tensorflow placeholder for target data
y = tf.placeholder(tf.int32, [None, max_length])
## Tensorflow placeholder for album lengths
l = tf.placeholder(tf.int32, [None])

## A recurrent neural network model, defined in model.py
m = dRNN(x, y, l, config())

## A Tensorflow variable initializer
init = tf.global_variables_initializer()

## A Tensorflow saver
saver = tf.train.Saver(max_to_keep=2)

## A Tensorflow session
sess = tf.Session()

sess.run(init)

## An output file to write error values
file = open('error.csv', 'w')

## @{Vectors of all of the inputs, targets, and lengths from the training set
train_data, train_target, train_length = train_input.all(1.0)
## @}
## @{Vectors of all of the inputs, targets, and lengths from the test set
test_data, test_target, test_length = test_input.all(1.0)
## @}

for step in range(steps):
    ## @{Vectors of inputs, targets, and length from the current mini-batch
    data, target, length = train_input.next(config().batch_size,
                                            step/config().entropy_saturation)
    ## @}

    sess.run(m.optimize, {x:data, y:target, l:length})

    if step % 100 == 0:
        ## Error evaluated on the training set
        train_err = sess.run(m.error,
                             {x:train_data, y:train_target, l:train_length})
        ## Error evaluated on the test sets
        test_err = sess.run(m.error,
                             {x:test_data, y:test_target, l:test_length})
        file.write('{:3.10f}, {:3.10f}\n'.format(train_err, test_err))
        print('Epoch {:2d} - Error {:3.1f}%'.format(step, 100*test_err))

    if step % 1000 == 0:
        saver.save(sess, './my-model', global_step=step)

train_err = sess.run(m.error, {x:train_data, y:train_target, l:train_length})
test_err = sess.run(m.error, {x:test_data, y:test_target, l:test_length})

file.write('{:3.10f}, {:3.10f}'.format(train_err, test_err))
file.close()

print('Epoch {:2d} error {:3.1f}%'.format(step + 1, 100 * train_err))
print('Test set error {:3.1f}%'.format(100 * test_err))

