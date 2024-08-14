import sys

from openai import OpenAI

from .utils import remove_emojis


class Llm:
    def __init__(self, params=None):
        self.params = params or {}
        self.model_name = self.params.get('model_name', None)
        self.model_file = self.params.get('model_file', None)
        self.num_gpu_layers = self.params.get('num_gpu_layers', None)
        self.context_length = self.params.get('context_length', None)
        self.streaming_output = self.params.get('streaming_output', None)
        self.chat_format = self.params.get('chat_format', None)
        self.system_message = self.params.get('system_message', None)
        self.verbose = self.params.get('verbose', None)
        self.base_url = self.params.get('base_url', None)
        self.api_key = self.params.get('api_key', None)
        self.temperature = self.params.get('temperature', 0.6)
        self.top_p = self.params.get('top_p', 0.9)
        self.bot_name = self.params.get('bot_name', 'Aria')

        self.llm = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

        self.messages = [{"role": "system", "content": self.system_message}]

    def get_answer(self, ui, ap, tts, data):
        self.messages.append({"role": "user", "content": data})

        outputs = self.llm.chat.completions.create(
            model=self.model_name,
            messages=self.messages,
            stream=self.streaming_output,
            temperature=self.temperature,
            top_p=self.top_p
        )

        if self.streaming_output:
            # CPT
            llm_output = ""
            tts_text_buffer = []
            color_code_block = False
            backticks = 0
            skip_code_block_on_tts = False
            ui.add_message(self.bot_name, "", new_entry=True)
            for i, out in enumerate(outputs):
                if "content" in out['choices'][0]["delta"]:
                    output_chunk_txt = out['choices'][0]["delta"]['content']
                    if output_chunk_txt == "``" and backticks == 0:
                        skip_code_block_on_tts = not skip_code_block_on_tts
                        color_code_block = not color_code_block
                        backticks += 2
                    if backticks > 0 and backticks <= 3:
                        backticks += 1
                    else:
                        backticks = 0
                    if i == 1:
                        print(self.bot_name + ':', output_chunk_txt.strip(), end='')
                        if backticks == 0:
                            ui.add_message(self.bot_name, output_chunk_txt.strip(), new_entry=False, color_code_block=color_code_block)
                    else:
                        print(output_chunk_txt, end='')
                        if backticks == 0:
                            ui.add_message(self.bot_name, output_chunk_txt, new_entry=False, color_code_block=color_code_block)
                    sys.stdout.flush()
                    llm_output += output_chunk_txt
                    if not skip_code_block_on_tts:
                        tts_text_buffer.append(output_chunk_txt)
                        if tts_text_buffer[-1] in [".", "!", "?", ":", "..", "..."]:
                            # TODO handle float numbers
                            # TODO remove multi dots
                            # TODO handle emphasis
                            txt_for_tts = remove_emojis("".join(tts_text_buffer).strip())
                            if len(txt_for_tts) > 1 and not all(char.isspace() for char in txt_for_tts):
                                tts.run_tts(txt_for_tts)
                            tts_text_buffer = []
            if not skip_code_block_on_tts and len(tts_text_buffer) != 0:
                # TODO remove multi dots
                txt_for_tts = remove_emojis("".join(tts_text_buffer).strip())
                if len(txt_for_tts) > 1 and not all(char.isspace() for char in txt_for_tts):
                    tts.run_tts(txt_for_tts)
            ap.check_audio_finished()
            print()
            llm_output = llm_output.strip()
        else:
            llm_output = outputs.choices[0].message.content.strip()

        self.messages.append({"role": "assistant", "content": llm_output})

        return llm_output
