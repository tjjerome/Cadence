import gpflow
import tensorflow as tf
import numpy as np

class LinearMulticlass(gpflow.models.Model):
    def __init__(self, X, Y, name=None):
        super().__init__(name=name) # always call the parent constructor

        self.X = X.copy() # X is a numpy array of inputs
        self.Y = Y.copy() # Y is a 1-of-k representation of the labels

        self.num_data, self.input_dim = X.shape
        _, self.num_classes = Y.shape

        #make some parameters
        self.W = gpflow.Param(np.random.randn(self.input_dim, self.num_classes))
        self.b = gpflow.Param(np.random.randn(self.num_classes))

        # ^^ You must make the parameters attributes of the class for
        # them to be picked up by the model. i.e. this won't work:
        #
        # W = gpflow.Param(...    <-- must be self.W

    @gpflow.params_as_tensors
    def _build_likelihood(self): # takes no arguments
        p = tf.nn.softmax(tf.matmul(self.X, self.W) + self.b) # Param variables are used as tensorflow arrays.
        return tf.reduce_sum(tf.log(p) * self.Y) # be sure to return a scalar

X = np.vstack([np.random.randn(10,2) + [2,2],
               np.random.randn(10,2) + [-2,2],
               np.random.randn(10,2) + [2,-2]])
Y = np.repeat(np.eye(3), 10, 0)

