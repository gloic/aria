import requests

class Tts:
    def __init__(self, params=None):
        self.params = params or {}
        self.api_url = self.params.get('api_url', 'http://localhost:8020/tts_stream')
        self.language = self.params.get('language', 'fr')
        self.voice_to_clone = self.params.get('assets', {}).get('voice_to_clone')

    def run_tts(self, nw, data):
        if not all(char.isspace() for char in data):
            params = {
                'text': data,
                'speaker_wav': self.voice_to_clone,
                'language': self.language
            }
            response = requests.get(self.api_url, params=params, stream=True)

            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    nw.receive_ack()
                    nw.send_msg(str(len(chunk)))
                    nw.receive_ack()
                    nw.send_audio(chunk)
        nw.receive_ack()
        nw.send_msg("tts_end")