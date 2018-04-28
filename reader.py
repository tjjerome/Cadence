## @package reader
#  Contains instructions on how to read the dataset

import numpy as np
import pickle
import sys
import util

from random import shuffle

## An object to hold the data in memory
class datastream(object):
    ## The constructor
    #  @param inputfile The name of the file to read
    #  @param config An object containing configuration variables defined in config.py
    def __init__(self, inputfile, config):
        ## input file object
        fin = open(inputfile, 'rb')
        ## A list to read albums into
        albums = []
        ## A list of album lengths
        self.lengths = []
        ## An empty array to store targets in
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
    
        ## The number of features in the song vector
        self.num_features = len(songs[0])
        ## An index to keep track of where to pull mini-batches from
        self.batch_id = 0
        ## Holds the album data
        self.album_stream = np.array(albums)
        fin.close()
        
    ## Returns a mini-batch of data
    #  @param batch_size The mini-batch size
    #  @param entropy The degree of which to shuffle the data
    #  @returns An array of shuffled albums
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

    ## Returns all of the data
    #  @param entropy The degree of which to shuffle the data
    #  @returns An array containing all of the albums in the dataset
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

    ## Gets parameters to help normalize the data
    #  @returns A vector of means for each song feature, A vector of standard deviations for each song feature
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

    ## Subtracts the mean from each element in the dataset and divides each by the standard deviations
    def normalize(self, mean, std):
        for i in range(len(self.album_stream)):
            for j in range(self.lengths[i]):
                for k in range(self.num_features):
                    self.album_stream[i, j, k] = (
                        self.album_stream[i, j, k] - mean[k]) / std[k]
