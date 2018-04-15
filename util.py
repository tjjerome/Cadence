import functools
import tensorflow as tf

def define_scope(function):
    attribute = '_cache_' + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(function.__name__):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator

# Helper functions

def str2int(s):
    chars = "0123456789qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM"
    i = 0
    for c in reversed(s):
        i *= len(chars)
        i += chars.index(c)
    return i % 1000003

def get_song_vector(song):
    song.pop('id',None)
    vec = list(song.items())
    vec.sort()
    for i in range(len(vec)):
        vec[i] = float(vec[i][1])
    return vec
