import gpflow
import numpy as np
import tensorflow as tf

f = open('kaggle_visible_evaluation_triplets.txt', 'r')

song_to_count = dict()

for line in f:
    _, song, _ = line.strip().split('\t')
    if song in song_to_count:
        song_to_count[song] += 1
    else:
        song_to_count[song] = 1

f.close()

songs_ordered = sorted(song_to_count.keys(), key=lambda s: song_to_count[s], reverse=True)

f = open('kaggle_visible_evaluation_triplets.txt', 'r')

user_to_songs = dict()

for line in f:
    user, song, _ =line.strip().split('\t')
    if user in user_to_songs:
        user_to_songs[user].add(song)
    else:
        user_to_songs[user] = set([song])

f.close()
