## @package config
#  Contains configuration options for the reader and network

## Object to return configuration variables
class config(object):
  ## The maximum length of albums in the dataset.
  #  This is the max that the Spotify API currently allows.
  #  Do not change unless the API updates.
  max_length = 49 
  ## The number of albums to train on at one time
  batch_size = 300
  ## The number of optimization loops to run
  training_steps = 1000000
  ## The step when the input reaches maximum entropy.
  #  Should be no greater than training_steps / 4
  entropy_saturation = 12000
  ## The number of nodes in the RNN
  hidden_size = 1000
  ## Learning rate of the Adam optimizer
  learning_rate = 0.001
  ## Probability of keeping each node through the dropout mask applied at each step
  keep_prob = 0.5
