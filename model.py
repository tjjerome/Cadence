import tensorflow as tf
from util import define_scope

class dRNN:

    def __init__(self, data, target, config):
        self.data = data
        self.target = target
        self.hidden_size = config.hidden_size
        self.learning_rate = config.learning_rate
        self.prediction
        self.optimize
        self.error
        self.cell = None

    @define_scope
    def prediction(self):
        data_size = int(self.data.get_shape()[1])
        num_features = int(self.data.get_shape()[2])

        x = tf.unstack(self.data, data_size, 1)
        
        w = tf.Variable(tf.random_uniform([self.hidden_size,
                                                num_features]))
        b = tf.Variable(tf.constant(1.0, shape=[num_features]))
        
        cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_size)
        
        outputs, states = tf.contrib.rnn.static_rnn(cell, x,
                                                    dtype=tf.float32)
        
        outputs = tf.stack(outputs)
        outputs = tf.transpose(outputs, [1,0,2])

        outputs = tf.reshape(outputs, [-1, self.hidden_size])

        return tf.nn.xw_plus_b(outputs, w, b)

    @define_scope
    def optimize(self):
        num_features = int(self.data.get_shape()[2])
        logits = tf.reshape(self.prediction, [1, -1, num_features])
        loss = tf.losses.mean_squared_error(self.target, logits)
        
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)

        return optimizer.minimize(loss)

    @define_scope
    def error(self):
        mistakes = tf.not_equal(self.target, self.prediction)

        return tf.reduce_mean(tf.cast(mistakes, tf.float32))
