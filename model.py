## @package model
#  Sets up the tensorflow graph

import tensorflow as tf
from util import define_scope

## Recursive neural network class
class dRNN:
    
    ## The constructor
    #  @param data The input data
    #  @param target The targets to train to
    #  @param length A vector of album lengths
    #  @param config An object with configuration variables defined in config.py
    def __init__(self, data, target, length, config):
        ## Stores input data
        self.data = data
        ## Stores the targets
        self.target = target
        ## Stores the length vector
        self.length = length
        ## Stores the dropout probability from the config
        self.keep_prob = config.keep_prob
        ## Stores the learning rate from the config
        self.rate = config.learning_rate
        ## Stores the hidden size from the config
        self.hidden_size = config.hidden_size
        ## Initializes the prediction function with the Tensorflow session
        self.prediction
        ## Initializes the optimize function with the Tensorflow session
        self.optimize
        ## Initializes the error function with the Tensorflow session
        self.error

    ## Runs the LSTM cells over the input data
    #  @returns a (batch_size, max_length, max_length) tensor with prediction values
    @define_scope
    def prediction(self):
        ## The max length of the albums in the mini batch
        data_size = int(self.data.get_shape()[1])
        ## The size of the target array
        target_size = int(self.target.get_shape()[1])
        ## The number of features in each song vector
        num_features = int(self.data.get_shape()[2])

        ## Weights for the forward pass
        w_fw = tf.Variable(tf.random_uniform([self.hidden_size,
                                           target_size]),
                        trainable=True,
                        name='weights')
        ## Bias for the forward pass
        b_fw = tf.Variable(tf.constant(1.0, shape=[target_size]),
                        trainable=True,
                        name='bias')
        
        ## The LSTM cell for the forward pass
        cell_fw = tf.contrib.rnn.LSTMCell(self.hidden_size,
                                       state_is_tuple=True)

        ## Weights for the backward pass
        w_bw = tf.Variable(tf.random_uniform([self.hidden_size,
                                           target_size]),
                        trainable=True,
                        name='weights')
        ## Bias for the backward pass
        b_bw = tf.Variable(tf.constant(1.0, shape=[target_size]),
                        trainable=True,
                        name='bias')
        
        ## The LSTM cell for the backward pass
        cell_bw = tf.contrib.rnn.LSTMCell(self.hidden_size,
                                       state_is_tuple=True)

        if self.keep_prob < 0.99:
            cell_fw = tf.contrib.rnn.DropoutWrapper(
                cell_fw, output_keep_prob=self.keep_prob,
                variational_recurrent=True,
                dtype=tf.float32)
            cell_bw = tf.contrib.rnn.DropoutWrapper(
                cell_bw, output_keep_prob=self.keep_prob,
                variational_recurrent=True,
                dtype=tf.float32)
        
        outputs, states = tf.nn.bidirectional_dynamic_rnn(
            cell_fw, cell_bw, self.data,
            sequence_length=self.length,
            dtype=tf.float32)

        output_fw, output_bw = outputs
        
        output_fw = tf.reshape(output_fw, [-1, self.hidden_size])
        output_bw = tf.reshape(output_bw, [-1, self.hidden_size])

        output = tf.nn.softmax(tf.add(tf.nn.xw_plus_b(output_fw, w_fw, b_fw),
                                      tf.nn.xw_plus_b(output_bw, w_bw, b_bw)))

        return tf.reshape(output, [-1, data_size, target_size])
        
    ## Defines the loss function and the optmizer operation
    @define_scope
    def optimize(self):
        loss = tf.contrib.seq2seq.sequence_loss(
            self.prediction,
            self.target,
            tf.sequence_mask(self.length,
                             self.data.get_shape()[1],
                             tf.float32),
            average_across_timesteps = False,
            average_across_batch = False)
        
        optimizer = tf.train.AdamOptimizer(self.rate)

        return optimizer.minimize(loss)

    ## Defines the error used to measure network performance
    #  @returns The mean value of mistakes over the dataset
    @define_scope
    def error(self):
        ## Remembers the dropout probability to turn back on after the error test
        kp = self.keep_prob

        # turn off the dropout
        self.keep_prob = 1
        mistakes = tf.cast(tf.not_equal(self.target,
                                        tf.argmax(self.prediction,
                                                  2,
                                                  output_type=tf.int32)),
                           tf.float32)

        # turn dropout back on
        self.keep_prob = kp
        
        # Ignore padded cells
        mask = tf.sequence_mask(self.length,
                                self.data.get_shape()[1],
                                tf.int32)
        padded, mistakes = tf.dynamic_partition(mistakes, mask, 2)
        
        return tf.reduce_mean(mistakes)
