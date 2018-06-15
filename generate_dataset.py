## @package generate_dataset
#  Pulls album data from the Spotify API and randomly partitions it into test and training datasets.
#  Takes a playlist URI as an argument to get a list of artists to pull albums from.
#  Usage: % python3 generate_dataset.py \<Spotify Playlist URI\>

from __future__ import print_function    # (at top of module)
from spotipy.oauth2 import SpotifyClientCredentials
import pickle
import spotipy
import sys
import numpy as np

## Returns up to 100 artists in the given playlist and related artists if there is room
#  @param uri A Spotify playlist URI
#  @returns A list of artist ids
def get_artists(uri):
    username = uri.split(':')[2]
    playlist_id = uri.split(':')[4]
    results = sp.user_playlist_tracks(username, playlist_id)['items']
    unique = set()
    full = False
    ids = []
    for track in results:
        artists = track['track']['artists']
        for artist in artists:
            id = artist['id']

            # Skip duplicates
            if not id in unique:
                ids.append(artist['id'])
                unique.add(id)
                if len(unique) >= 200:
                    full = True
                else:
                    related_artists = sp.artist_related_artists(id)['artists']
                    artists.extend(related_artists)

            if full: break
        if full: break
    return ids

## Returns track ids for a given album
#  @param album A Spotify album id
#  @returns A list of track ids
def get_album_tracks(album):
    tracks = []
    ids = []
    results = sp.album_tracks(album)
    tracks.extend(results['items'])
    while results['next']:
        results = sp.next(results)
        tracks.extend(results['items'])
    for track in tracks:
        ids.append(track['id'])
    return ids

## Returns a list of album ids produced by a given artist
#  @param artist A Spotify artist id
#  @returns A list of album ids
def get_artist_albums(artist):
    albums = []
    ids = []
    results = sp.artist_albums(artist, album_type='album')
    albums.extend(results['items'])
    while results['next']:
        results = sp.next(results)
        albums.extend(results['items'])
    unique = set()  # skip duplicate albums
    for album in albums:
        name = album['name'].lower()
        if not name in unique:  
            ids.append(album['id'])
            unique.add(name)
    return ids

###########################################################################

## Gets Spotify credentials from the source environment
client_credentials_manager = SpotifyClientCredentials()
## Initializes a Spotify object with the credentials
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

if len(sys.argv) > 1:
    ## A Spotify playlist URI
    uri = sys.argv[1]
else:
    uri = 'spotify:user:1230457813:playlist:74oqUs80qzfsdCr0Ek9tZV'

## A list of artist ids from the given playlist
artists = get_artists(uri)

## An output file object
fout = open('data', 'wb')

for artist in artists:

    ## A list of album ids
    albums = get_artist_albums(artist)

    for album in albums:
        ## Indicates whether the album had any track with missing information
        bad_track = False
        ## A list of track ids
        results = get_album_tracks(album)
        if len(results) >= 50: continue
        ## A list of json objects with song features
        tracks = sp.audio_features(results)
        
        for track in tracks:
            if track:
                track.pop('type', None)
                track.pop('track_href', None)
                track.pop('analysis_url', None)
                track.pop('uri', None)
                if len(track) != 14: bad_track = True
            else:
                bad_track = True
            
        # toss the album if there is a bad track
        if bad_track: continue
        pickle.dump(tracks, fout)

fout.close()

## An input file object
fin = open('data', 'rb')
## An output file object for the test dataset
test = open('test', 'wb')
## An output file object for the training dataset
train = open('train', 'wb')

## Counts the number of albums processed
n = 0
## Counts the number of albums in the training dataset
train_n = 0
## Counts the number of albums in the test dataset
test_n = 0

while True:
    try:
        if np.random.binomial(1,0.1) == 0:
            pickle.dump(pickle.load(fin), train)
            train_n += 1
            
        else:
            pickle.dump(pickle.load(fin), test)
            test_n += 1
            
        n += 1
        
    except EOFError:
        break

print('{} albums processed'.format(n))
print('Train - {}, Test - {}'.format(train_n, test_n))
        
fin.close()
test.close()
train.close()
