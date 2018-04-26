from __future__ import print_function    # (at top of module)
from spotipy.oauth2 import SpotifyClientCredentials
import pickle
import spotipy
import sys
import numpy as np

#Hot Fuss
#spotify:album:4undIeGmofnAYKhnDclN1w

#The Killers
#spotify:artist:0C0XlULifJtAgn6ZNCW2eu

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
            if not id in unique:
                ids.append(artist['id'])
                unique.add(id)
                if len(unique) > 499:
                    full = True
                else:
                    related_artists = sp.artist_related_artists(id)['artists']
                    artists.extend(related_artists)

            if full: break
        if full: break
    return ids

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

client_credentials_manager = SpotifyClientCredentials()
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

if len(sys.argv) > 1:
    uri = sys.argv[1]
else:
    uri = 'spotify:user:1230457813:playlist:74oqUs80qzfsdCr0Ek9tZV'

artists = get_artists(uri)

fout = open('data', 'wb')

for artist in artists:

    albums = get_artist_albums(artist)

    for album in albums:
        bad_track = False
        results = get_album_tracks(album)
        if len(results) >= 50: continue
        tracks = sp.audio_features(results)
        
        for track in tracks:
            if track:
                track.pop('type', None)
                track.pop('track_href', None)
                track.pop('analysis_url', None)
                track.pop('uri', None)
            if len(track) != 14: bad_track = True
            
        if bad_track: continue
        pickle.dump(tracks, fout)

fout.close()

fin = open('data', 'rb')
test = open('test', 'wb')
train = open('train', 'wb')

n = 0
train_n = 0
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
