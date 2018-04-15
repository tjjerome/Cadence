import numpy as np
import pickle
import sys
import util

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
            
        albums = (self.album_stream[self.batch_id:min(
            self.batch_id + batch_size,
            len(self.album_stream))])
        albumlen = (self.lengths[self.batch_id:min(
            self.batch_id + batch_size,
            len(self.lengths))])
        self.batch_id = min(self.batch_id + batch_size,
                            len(self.album_stream))
        return albums, albumlen
