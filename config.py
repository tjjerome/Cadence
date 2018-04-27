class config(object):
  max_length = 49 #This is the max that the Spotify API currently allows
                  #Do not change unless the API updates
  batch_size = 300
  training_steps = 250000
  entropy_saturation = 62500
  hidden_size = 1000
  learning_rate = 0.001
  keep_prob = 0.5
