## @package util
#  Contains useful functions

import functools
import tensorflow as tf

## A decorator function to set scopes on Tensorflow variables
#  @param function The function to set scopes on
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

## Converts a string to an int by treating it like a base-62 number
#  @param s The input string
#  @returns An int representing the string
def str2int(s):
    chars = "0123456789qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM"
    i = 0
    for c in reversed(s):
        i *= len(chars)
        i += chars.index(c)
    return i % 1000003

## Converts a json object containing song features to a vector
#  @param song The input song
#  @returns The output vector
def get_song_vector(song):
    song.pop('id', None)
    vec = list(song.items())
    vec.sort()
    for i in range(len(vec)):
        vec[i] = float(vec[i][1])
    return vec
