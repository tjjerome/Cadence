import tensorflow as tf
from util import define_scope

class dRNN:

    def __init__(self, data, target, length, config):
        self.data = data
        self.target = target
        self.length = length
        #self.global_step = tf.Variable(tf.constant(0, dtype=tf.int64))
        self.keep_prob = config.keep_prob
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

        w_fw = tf.Variable(tf.random_uniform([self.hidden_size,
                                           target_size]),
                        trainable=True,
                        name='weights')
        b_fw = tf.Variable(tf.constant(1.0, shape=[target_size]),
                        trainable=True,
                        name='bias')
        
        cell_fw = tf.contrib.rnn.LSTMCell(self.hidden_size,
                                       state_is_tuple=True)

        w_bw = tf.Variable(tf.random_uniform([self.hidden_size,
                                           target_size]),
                        trainable=True,
                        name='weights')
        b_bw = tf.Variable(tf.constant(1.0, shape=[target_size]),
                        trainable=True,
                        name='bias')
        
        cell_bw = tf.contrib.rnn.LSTMCell(self.hidden_size,
                                       state_is_tuple=True)

        if self.keep_prob < 0.99:
            cell_fw = tf.contrib.rnn.DropoutWrapper(
                cell_fw, output_keep_prob=self.keep_prob)
            cell_bw = tf.contrib.rnn.DropoutWrapper(
                cell_bw, output_keep_prob=self.keep_prob)
        
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
        kp = self.keep_prob
        self.keep_prob = 1
        mistakes = tf.cast(tf.not_equal(self.target,
                                        tf.argmax(self.prediction,
                                                  2,
                                                  output_type=tf.int32)),
                           tf.float32)

        self.keep_prob = kp
        
        # Ignore padded cells
        mask = tf.sequence_mask(self.length,
                                self.data.get_shape()[1],
                                tf.int32)
        padded, mistakes = tf.dynamic_partition(mistakes, mask, 2)
        
        return tf.reduce_mean(mistakes)
