import numpy as np
import pickle
import sys
import util

from random import shuffle

class datastream(object):
    def __init__(self, inputfile, config):
        fin = open(inputfile, 'rb')
        albums = []
        self.lengths = []
        self.targets = np.array([]).reshape(0, config.max_length)
        
        while True:
            try:
                album = pickle.load(fin)
                songs = []
                song_stream = []
                target = np.zeros((1, config.max_length))

                # Convert each song to a vector representation
                for song in album:
                    songs.append(util.get_song_vector(song))

                # Keep track of the length of each album
                self.lengths.append(len(songs))

                # Create a shuffling index, skip the first one
                indices = np.arange(1, len(songs))
                shuffle(indices)
                song_stream.append(songs[0])

                # Shuffle the album by the index and keep track of the indeces
                for i in range(len(songs)-1):
                    song_stream.append(songs[indices[i]])
                    target[0, indices[i]] = i+1

                # Pad the array with zeros
                for i in range(config.max_length - len(song_stream)):
                    song_stream.append([0]*len(song_stream[0]))
                
                albums.append(song_stream)
                self.targets = np.vstack((self.targets, target))
            except EOFError:
                break
    
        self.num_features = len(song_stream[0])
        self.batch_id = 0
        self.album_stream = np.array(albums)
        fin.close()
        
    def next(self, batch_size=1):
        if self.batch_id == len(self.album_stream):
            self.batch_id = 0
            
        data = self.album_stream[self.batch_id:min(
            self.batch_id + batch_size,
            len(self.album_stream))]
        target = self.targets[self.batch_id:min(
            self.batch_id + batch_size,
            len(self.album_stream))]
        length = np.array(self.lengths[self.batch_id:min(
            self.batch_id + batch_size,
            len(self.lengths))])
            
        self.batch_id = min(self.batch_id + batch_size,
                            len(self.album_stream))
        return data, target, length

    def all(self):
        data = self.album_stream
        
        target = self.targets

        length = np.array(self.lengths)
    
        return data, target, length

    def norm_params(self):
        avg = []
        var = []

        # Get mean and variance within each album
        for i in range(len(self.album_stream)):
            a = np.array(self.album_stream[i])
            avg.append(np.mean(a[:self.lengths[i],:],
                               axis=0))
            var.append(np.var(a[:self.lengths[i],:],
                              axis=0))

        # Get mean an std for dataset
        mean = np.mean(avg, axis=0)
        std = np.sqrt(np.mean(var, axis=0))

        return mean, std

    def normalize(self, mean, std):
        for i in range(len(self.album_stream)):
            for j in range(self.lengths[i]):
                for k in range(self.num_features):
                    self.album_stream[i, j, k] = (
                        self.album_stream[i, j, k] - mean[k]) / std[k]