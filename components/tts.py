import requests
import numpy as np
import io
import wave

class Tts:
    def __init__(self, params=None, ap=None):
        self.params = params or {}
        self.api_url = self.params.get('api_url', None)
        self.language = self.params.get('language', None)
        self.voice_to_clone = self.params.get('assets', {}).get('voice_to_clone')
        self.ap = ap
        self.device = self.params.get('device', None)

    def run_tts(self, data):
        if not all(char.isspace() for char in data):
            params = {
                'text': data,
                'speaker_wav': self.voice_to_clone,
                'language': self.language
            }
        response = requests.get(f"{self.api_url}/tts_stream", params=params, stream=True)

        if response.status_code != 200:
            raise Exception(f"Erreur lors de l'appel à l'API: {response.status_code}")

        wav_header = response.raw.read(44)  # L'en-tête WAV fait généralement 44 octets
        wav_file = wave.open(io.BytesIO(wav_header), 'rb')
        sample_width = wav_file.getsampwidth()
        channels = wav_file.getnchannels()
        frame_rate = wav_file.getframerate()

        # Configurer le lecteur audio avec les informations de l'en-tête WAV si nécessaire
        # self.ap.configure_audio(sample_width, channels, frame_rate)

        # Lire les chunks audio
        chunk_size = 100
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                element_size = np.dtype(np.int16).itemsize
                remainder = len(chunk) % element_size
                if remainder != 0:
                    chunk += b'\x00' * (element_size - remainder)

                audio_array = np.frombuffer(chunk, dtype=np.int16)
                audio_array = audio_array.astype(np.float32)
                self.ap.stream_sound(audio_array, update_ui=True)

        return 'tts_done'
