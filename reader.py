import numpy as np
import pickle
import sys
import util

from random import shuffle

class datastream(object):
    def __init__(self, inputfile, config):
        fin = open(inputfile, 'rb')
        self.album_stream = []
        self.lengths = []
        self.targets = np.array([]).reshape(0,config.max_length,config.max_length)
        
        while True:
            try:
                album = pickle.load(fin)
                songs = []
                song_stream = []
                target = np.zeros((1, config.max_length, config.max_length))

                # Convert each song to a vector representation
                for song in album:
                    songs.append(util.get_song_vector(song))

                # Keep track of the length of each album
                self.lengths.append(len(songs))

                # Create a shuffling index
                indeces = np.arange(len(songs))
                shuffle(indeces)

                # Shuffle the album by the index and keep track of the indeces
                for i in range(len(songs)):
                    song_stream.append(songs[indeces[i]])
                    target[0, i, len(songs):] = 0.1 # differ from padding
                    target[0, i, indeces[i]] = 1.0

                # Pad the array with zeros
                for i in range(config.max_length - len(song_stream)):
                    song_stream.append([0]*len(song_stream[0]))
                
                self.album_stream.append(song_stream)
                self.targets = np.vstack((self.targets, target))
            except EOFError:
                break
    
        self.num_features = len(song_stream[0])
        self.batch_id = 0
        fin.close()
        
    def next(self, batch_size=1):
        if self.batch_id == len(self.album_stream):
            self.batch_id = 0
            
        data = np.array(self.album_stream[self.batch_id:min(
            self.batch_id + batch_size,
            len(self.album_stream))])
        length = np.array(self.lengths[self.batch_id:min(
            self.batch_id + batch_size,
            len(self.lengths))])
        target = self.targets[self.batch_id:min(
            self.batch_id + batch_size,
            len(self.album_stream))]
            
        self.batch_id = min(self.batch_id + batch_size,
                            len(self.album_stream))
        return data, target, length

    def all(self):
        data = np.array(self.album_stream)
        
        target = self.targets

        length = np.array(self.lengths)
    
        return data, target, length
