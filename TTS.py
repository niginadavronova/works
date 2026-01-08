import numpy as np
from transformers import AutoProcessor, BarkModel
import os
import scipy
import nltk
from nltk.tokenize import sent_tokenize
import torch
from bark import SAMPLE_RATE, generate_audio, preload_models



nltk.download('punkt')
model_folder = './suno_bark_model'
# if os.path.exists(model_folder):
#     print('Loading the model from local directory...')
#     processor = AutoProcessor.from_pretrained(model_folder)
#     model = BarkModel.from_pretrained(model_folder)
# else:
#     print('Downloading the model from huggingface...')
#     processor = AutoProcessor.from_pretrained("suno/bark-small")
#     model = BarkModel.from_pretrained("suno/bark-small")
#     model.save_pretrained(model_folder)
#     processor.save_pretrained(model_folder)

voice_preset = 'v2/ru_speaker_5'
# inputs = processor("Привет, уважаемые студенты! [laughs] Как ваши дела?", voice_preset=voice_preset)
# audio = model.generate(**inputs)
# audio = audio.cpu().numpy().squeeze()
# scipy.io.wavfile.write('test.wav',data=audio, rate=SAMPLE_RATE)
# #
# #
# def generate_audio(text, voice):
#     inputs = processor(text, voice_preset=voice, return_tensors='pt')
#     audio = model.generate(**inputs)
#     audio = audio.cpu().numpy().squeeze()
#     return audio
#
# text_input = 'Привет, уважаемые студенты! [laughs] Как ваши дела?  Привет, уважаемые студенты! [laughs] Как ваши дела?  Привет, уважаемые студенты! [laughs] Как ваши дела?'
#
# if len(text_input) >= 100:
#     print('Text is too long! Splitting into sentences...')
#     sentences = sent_tokenize(text_input, language='russian')
#     silence_duration = 0.25
#     silence_duration = np.zeros(int(silence_duration*SAMPLE_RATE))
#     parts = []
#     for part in sentences:
#         print('Processing:', part)
#         audio = generate_audio(text_input, voice_preset)
#         parts.extend([audio, silence_duration])  # []
#     full_audio = np.concatenate(parts)
# else:
#     print('Generating audio...')
#     full_audio = generate_audio(text_input, voice_preset)
#
#
# scipy.io.wavfile.write('test2.wav',data=full_audio.astype(np.float32()), rate=SAMPLE_RATE)
# print('Audio file is saved!')


class TTS_Generator():
    def __init__(self, voice='v2/ru_speaker_5', model_path='./suno_bark_model'):
        self.model_path = model_path
        self.voice = voice
        self.sample_rate = SAMPLE_RATE
        self.processor, self.model = self.load_model_processor()

    def load_model_processor(self, model_path='./suno_bark_model'):
        if os.path.exists(model_path):
            print('Loading the model from local directory...')
            processor = AutoProcessor.from_pretrained(model_path)
            model = BarkModel.from_pretrained(model_path)
        else:
            print('Downloading the model from huggingface...')
            processor = AutoProcessor.from_pretrained("suno/bark-small")
            model = BarkModel.from_pretrained("suno/bark-small")
            model.save_pretrained(model_path)
            processor.save_pretrained(model_path)

        return processor, model

    def generate_audio(self, text, voice):
        inputs = self.processor(text, voice_preset=voice, return_tensors='pt')
        audio = self.model.generate(**inputs)
        audio = audio.cpu().numpy().squeeze()
        return audio

    def audio_synthesis(self, text, voice, output_file='output.wav'):

        if len(text) >= 100:
            print('Text is too long! Splitting into sentences...')
            sentences = sent_tokenize(text, language='russian')
            silence_duration = 0.25
            silence_duration = np.zeros(int(silence_duration * SAMPLE_RATE))
            parts = []
            for part in sentences:
                print('Processing:', part)
                audio = self.generate_audio(text, voice)
                parts.extend([audio, silence_duration])  # []
            full_audio = np.concatenate(parts)
        else:
            print('Generating audio...')
            full_audio = self.generate_audio(text, voice)

        scipy.io.wavfile.write(output_file, data=full_audio.astype(np.float32()), rate=SAMPLE_RATE)
        print('Audio file is saved!')
