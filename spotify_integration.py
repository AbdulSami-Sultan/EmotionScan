from spotipy import Spotify
from spotipy.oauth2 import SpotifyOAuth

def get_spotify_recommendation(emotion):
    emotion_playlists = {
        0: "happy_playlist_id",
        1: "sad_playlist_id",
        2: "angry_playlist_id",
        3: "neutral_playlist_id",
    }

    sp = Spotify(auth_manager=SpotifyOAuth(client_id="YOUR_CLIENT_ID",
                                           client_secret="YOUR_CLIENT_SECRET",
                                           redirect_uri="http://localhost:8888/callback",
                                           scope="user-modify-playback-state"))
    playlist_id = emotion_playlists.get(emotion, "default_playlist_id")
    sp.start_playback(context_uri=f"spotify:playlist:{playlist_id}")

