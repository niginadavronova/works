from TTS.tts.configs.bark_config import BarkConfig
from TTS.tts.models.bark import Bark
import scipy

config = BarkConfig()
model = Bark.init_from_config(config)
model.load_checkpoint(config, checkpoint_dir="bark/", eval=True)

text = "This is a test of cloning voice."

output_dict = model.synthesize(
    text,
    config,
    speaker_id="speaker",
    voice_dirs="bark_voices/"
)

sample_rate = 24000
scipy.io.wavfile.write("generated_audio.wav", rate=sample_rate, data=output_dict["wav"])  # Access the "wav" key correctly
print("Audio file has been saved as 'generated_audio.wav'.")



# import scipy
# from TTS.tts.configs.bark_config import BarkConfig
# from TTS.tts.models.bark import Bark
#
# class VoiceCloner:
#     def __init__(self, config_dir='bark', voices_dir='bark_voices/'):
#         self.config = BarkConfig()
#         self.model = Bark.init_from_config(self.config)
#         self.model.load_checkpoint(self.config, checkpoint_dir=config_dir, eval=True)
#         self.voices_dir = voices_dir
#
#     def clone_voice(self, text, speaker_id="speaker", output_file="cloned_audio.wav"):
#         try:
#             output_dict = self.model.synthesize(text, self.config, speaker_id=speaker_id, voice_dirs=self.voices_dir)
#             sample_rate = 24000
#             scipy.io.wavfile.write(output_file, rate=sample_rate, data=output_dict["wav"])
#             return output_file, sample_rate
#         except Exception as e:
#             raise RuntimeError(f"Failed to generate voice: {e}")


import os
import numpy as np
import scipy
from TTS.tts.configs.bark_config import BarkConfig
from TTS.tts.models.bark import Bark


class VoiceCloner:
    def __init__(self, config_dir='bark', voices_dir='bark_voices/'):
        self.config = BarkConfig()
        self.model = Bark.init_from_config(self.config)
        self.model.load_checkpoint(self.config, checkpoint_dir=config_dir, eval=True)
        self.voices_dir = voices_dir
        if not os.path.exists(self.voices_dir):
            os.makedirs(self.voices_dir)

    def clone_voice(self, text, speaker_id="speaker", output_file="cloned_audio.wav"):
        try:
            output_dict = self.model.synthesize(text, self.config, speaker_id=speaker_id, voice_dirs=self.voices_dir)
            sample_rate = 24000
            scipy.io.wavfile.write(output_file, rate=sample_rate, data=output_dict["wav"])
            return output_file, sample_rate
        except Exception as e:
            raise RuntimeError(f"Failed to generate voice: {e}")

    def process_voice_sample(self, voice_file):

        try:
            speaker_embedding = self.model.extract_speaker_embedding(voice_file)
            return speaker_embedding
        except Exception as e:
            raise RuntimeError(f"Failed to process voice sample: {e}")

    def add_speaker_embedding(self, speaker_id, speaker_embedding):

        try:
            speaker_dir = os.path.join(self.voices_dir, speaker_id)
            if not os.path.exists(speaker_dir):
                os.makedirs(speaker_dir)

            embedding_file = os.path.join(speaker_dir, "embedding.npy")
            np.save(embedding_file, speaker_embedding)
        except Exception as e:
            raise RuntimeError(f"Failed to add speaker embedding: {e}")
