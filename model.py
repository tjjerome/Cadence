import tensorflow as tf
from util import define_scope

class dRNN:

    def __init__(self, data, target, length, config):
        self.data = data
        self.target = target
        self.length = length
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
                                           target_size]))
        b = tf.Variable(tf.constant(1.0, shape=[target_size]))
        
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
        #num_features = int(self.data.get_shape()[2])
        #logits = tf.reshape(self.prediction, [1, -1, num_features])
        loss = -tf.reduce_mean(self.target * tf.log(tf.clip_by_value(
            self.prediction, 1e-10, 1.0)))
        
        optimizer = tf.train.AdamOptimizer()

        return optimizer.minimize(loss)

    @define_scope
    def error(self):
        #mistakes = tf.not_equal(tf.argmax(self.target,1),
                                #tf.argmax(self.prediction,1))
        loss = -tf.reduce_sum(self.target * tf.log(tf.clip_by_value(
            self.prediction, 1e-10, 1.0)))
        return loss
        #return tf.reduce_mean(tf.cast(mistakes, tf.float32))
