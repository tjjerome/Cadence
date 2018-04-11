from spotipy.oauth2 import SpotifyClientCredentials
import spotipy
import json

#spotify:user:1230457813:playlist:74oqUs80qzfsdCr0Ek9tZV

client_credentials_manager = SpotifyClientCredentials()
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

uri = 'spotify:user:1230457813:playlist:74oqUs80qzfsdCr0Ek9tZV'
username = uri.split(':')[2]
playlist_id = uri.split(':')[4]
results = sp.user_playlist_tracks(username, playlist_id)['items']
unique = set()
ids = []
for track in results:
    artists = track['track']['artists']
    for artist in artists:
        id = artist['id']
        if not id in unique:
            ids.append(artist['id'])
            unique.add(id)

print(ids)
