import requests
import numpy as np

class Tts:
    def __init__(self, params=None):
        self.params = params or {}
        self.api_url = self.params.get('api_url', None)
        self.language = self.params.get('language', None)
        self.voice_to_clone = self.params.get('assets', {}).get('voice_to_clone')
        self.chunk_size = 100
    def call_api(self, text):
        params = {
            'text': text,
            'speaker_wav': self.voice_to_clone,
            'language': self.language,
            'chunk_size': self.chunk_size
        }
        response = requests.get(f"{self.api_url}/tts_stream", params=params, stream=True)
        response.raise_for_status()
        return response

    def run_tts(self, nw, data):
        if not all(char.isspace() for char in data):
            response = self.call_api(data)
            for chunk in response.iter_content(chunk_size=self.chunk_size):
                if chunk:
                    audio_chunk = np.frombuffer(chunk, dtype=np.float32)
                    nw.receive_ack()
                    nw.send_msg(str(len(audio_chunk.tobytes())))
                    nw.receive_ack()
                    nw.send_audio(audio_chunk.tobytes())
        nw.receive_ack()
        nw.send_msg("tts_end")
        return 'tts_done'
