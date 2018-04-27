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
                target = np.zeros((1, config.max_length))

                # Convert each song to a vector representation
                for song in album:
                    songs.append(util.get_song_vector(song))

                # Keep track of the length of each album
                self.lengths.append(len(songs))

                # Fill target array with album indices
                for i in range(len(songs)):
                    target[0, i] = i

                # Pad the array with zeros
                for i in range(config.max_length - len(songs)):
                    songs.append([0]*len(songs[0]))
                
                albums.append(songs)
                self.targets = np.vstack((self.targets, target))
            except EOFError:
                break
    
        self.num_features = len(songs[0])
        self.batch_id = 0
        self.album_stream = np.array(albums)
        fin.close()
        
    def next(self, batch_size=1, entropy=0.0):
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

        if entropy > 0.0:
            if entropy > 1.0: entropy = 1.0
            for i in range(len(data)):
                num_swaps = int(entropy*(length[i]-2))
                indices = np.arange(1, length[i]-1)
                shuffle(indices)
                more_indices = np.arange(1, length[i]-1)
                shuffle(more_indices)
                indices = np.append(indices, more_indices)
                for j in range(num_swaps):
                    index1 = indices[2*j]
                    index2 = indices[2*j+1]
                    data[i][[index1, index2]] = data[i][[index2, index1]]
                    target[i][[index1, index2]] = target[i][[index2, index1]]
                 
        self.batch_id = min(self.batch_id + batch_size,
                            len(self.album_stream))
        return data, target, length

    def all(self, entropy=0.0):
        data = self.album_stream
        
        target = self.targets

        length = np.array(self.lengths)
        
        if entropy > 0.0:
            if entropy > 1.0: entropy = 1.0
            for i in range(len(data)):
                num_swaps = int(entropy*(length[i]-2))
                indices = np.arange(1, length[i]-1)
                shuffle(indices)
                more_indices = np.arange(1, length[i]-1)
                shuffle(more_indices)
                indices = np.append(indices, more_indices)
                for j in range(num_swaps):
                    index1 = indices[2*j]
                    index2 = indices[2*j+1]
                    data[i][[index1, index2]] = data[i][[index2, index1]]
                    target[i][[index1, index2]] = target[i][[index2, index1]]
                    
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

        # Get mean and std for dataset
        mean = np.mean(avg, axis=0)
        std = np.sqrt(np.mean(var, axis=0))

        return mean, std

    def normalize(self, mean, std):
        for i in range(len(self.album_stream)):
            for j in range(self.lengths[i]):
                for k in range(self.num_features):
                    self.album_stream[i, j, k] = (
                        self.album_stream[i, j, k] - mean[k]) / std[k]
