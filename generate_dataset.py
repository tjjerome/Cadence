from __future__ import print_function    # (at top of module)
from spotipy.oauth2 import SpotifyClientCredentials
import json
import pickle
import spotipy
import sys

#Hot Fuss
#spotify:album:4undIeGmofnAYKhnDclN1w

#The Killers
#spotify:artist:0C0XlULifJtAgn6ZNCW2eu

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

def get_artist_albums(id):
    albums = []
    ids = []
    results = sp.artist_albums(id, album_type='album')
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

client_credentials_manager = SpotifyClientCredentials()
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

if len(sys.argv) > 1:
    id = sys.argv[1]
else:
    id = '0C0XlULifJtAgn6ZNCW2eu'

n = 0
    
albums = get_artist_albums(id)

fout = open('test', 'wb')

for aid in albums:
    features = sp.audio_features(get_album_tracks(aid))
    
    for track in features:
        del track['type']
        del track['track_href']
        del track['analysis_url']
        del track['uri']

    pickle.dump(features, fout)
    n += 1

fout.close()

print('{} albums processed'.format(n))
