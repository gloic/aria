import numpy as np
import requests


class Tts:
    def __init__(self, params=None, ap=None):
        self.params = params or {}
        self.api_url = self.params.get('api_url', None)
        self.language = self.params.get('language', None)
        self.voice_to_clone = self.params.get('assets', {}).get('voice_to_clone')
        self.ap = ap
        self.device = self.params.get('device', None)

    def run_tts(self, data):
        #if not all(char.isspace() for char in data):
        params = {
            'text': data,
            'speaker_wav': self.voice_to_clone,
            'language': self.language
        }
        response = requests.get(f"{self.api_url}/tts_stream", params=params, stream=True)

        if response.status_code != 200:
            raise Exception(f"API error: {response.status_code}")

        for chunk in response.iter_content(chunk_size=100):
            if chunk:
                chunk_array = np.frombuffer(chunk, dtype=np.int16).astype(np.float32)
                chunk_array = chunk_array / 32767.0

                self.ap.stream_sound(chunk_array, update_ui=True)

        return 'tts_done'
