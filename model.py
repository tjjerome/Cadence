import tensorflow as tf
from util import define_scope

class dRNN:

    def __init__(self, data, target, length, config):
        self.data = data
        self.target = target
        self.length = length
        self.rate = config.learning_rate
        self.hidden_size = config.hidden_size
        self.prediction
        self.optimize
        self.error

    @define_scope
    def prediction(self):
        data_size = int(self.data.get_shape()[1])
        target_size = int(self.target.get_shape()[1])
        num_features = int(self.data.get_shape()[2])

        w = tf.Variable(tf.random_uniform([self.hidden_size,
                                           target_size]),
                        trainable=True,
                        name='weights')
        b = tf.Variable(tf.constant(1.0, shape=[target_size]),
                        trainable=True,
                        name='bias')
        
        cell = tf.contrib.rnn.LSTMCell(self.hidden_size,
                                       state_is_tuple=True)
        
        outputs, states = tf.nn.dynamic_rnn(cell, self.data,
                                            sequence_length=self.length,
                                            dtype=tf.float32)

        #outputs = tf.transpose(outputs, [1,0,2])

        outputs = tf.reshape(outputs, [-1, self.hidden_size])

        #ten = tf.gather(outputs, 9)
        
        outputs = tf.nn.softmax(tf.nn.xw_plus_b(outputs, w, b))

        return tf.reshape(outputs, [-1, data_size, target_size])
        
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

    @define_scope
    def error(self):
        mistakes = tf.cast(tf.not_equal(self.target,
                                        tf.argmax(self.prediction,
                                                  2,
                                                  output_type=tf.int32)),
                           tf.float32)

        # Ignore padded cells
        mask = tf.sequence_mask(self.length,
                                self.data.get_shape()[1],
                                tf.int32)
        padded, mistakes = tf.dynamic_partition(mistakes, mask, 2)
        
        return tf.reduce_mean(mistakes)
