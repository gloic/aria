import requests
import numpy as np
import io
import soundfile as sf  # Vous aurez besoin de cette bibliothèque pour lire les fichiers audio


class Tts:
    def __init__(self, params=None, ap=None):
        self.params = params or {}
        self.api_url = self.params.get('api_url', 'http://localhost:8020/tts_stream')
        self.language = self.params.get('language', 'fr')
        self.voice_to_clone = self.params.get('assets', {}).get('voice_to_clone')
        self.ap = ap
        self.device = self.params.get('device', None)

    def run_tts(self, nw, data):
        if not all(char.isspace() for char in data):
            params = {
                'text': data,
                'speaker_wav': self.voice_to_clone,
                'language': self.language
            }
            response = requests.get(self.api_url, params=params, stream=True)

            audio_data = bytearray()
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    audio_data.extend(chunk)
            audio_array, sample_rate = sf.read(io.BytesIO(audio_data))

            if self.device == 'gpu':
                audio_array = np.ascontiguousarray(audio_array)
            else:
                audio_array = np.asarray(audio_array)

            self.ap.stream_sound(audio_array, update_ui=True)

        return 'tts_done'
