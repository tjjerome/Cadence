# Cadence
Organize playlists like an album using a recurrent neural network.

The program is currently UNFINISHED. It is still learning to unshuffle albums, which will eventually be applied to playlists to "unshuffle" them into a logical order.

## Dependencies

<a href="https://www.tensorflow.org/install/">Tensorflow v1.7.0 or greater</a>

<a href="http://spotipy.readthedocs.io/en/latest/#installation">Spotipy</a> - Only needed to generate new data

NumPy

## Usage

To use the program as is, simply clone the repo and run the following:

% python3 main.py

Edit parameters in config.py to adjust the training of the network.

## Generating data

Cadence will train on the albums of artists (and related artists) in the playlist selected by the user.
A small dataset is included in the repo for testing purposes, so it is not necessary to generate a new one.
If new data is desired, simply run the following in the repo directory:

% python3 generate_dataset.py \<Spotify Playlist URI\>

This requires the user to have proper credentials to access the Spotify API. These are not currently included in this repository.
 For details on how to obtain credentials, refer to the
 <a href="http://spotipy.readthedocs.io/en/latest/#authorized-requests">spotipy documentation</a>.
 
 ## NIPS Paper
 
 Found in /paper/
