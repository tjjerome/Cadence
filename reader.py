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
        while True:
            try:
                album = pickle.load(fin)
                song_stream = []
                
                for song in album:
                    song_stream.append(util.get_song_vector(song))
                self.lengths.append(len(song_stream))
                for i in range(config.max_length - len(song_stream)):
                    song_stream.append([0]*len(song_stream[0]))
                
                self.album_stream.append(song_stream)
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
        #albumlen = (self.lengths[self.batch_id:min(
            #self.batch_id + batch_size,
            #len(self.lengths))])
        target = data[:,:,4]

        data = np.swapaxes(data,0,1)
        shuffle(data)
        data = np.swapaxes(data,0,1)
            
        self.batch_id = min(self.batch_id + batch_size,
                            len(self.album_stream))
        return data, target

    def all(self):
        data = np.array(album_stream)
        
        target = data[:,:,4]

        data = np.swapaxes(data,0,1)
        shuffle(data)
        data = np.swapaxes(data,0,1)
    
        return data, target
