# Imports used through the rest of the notebook.
import os
import time

import IPython
import torchaudio

from tortoise.api import TextToSpeech
from tortoise.utils.audio import load_voice

# This will download all the models used by Tortoise from the HuggingFace hub.
tts = TextToSpeech()

# Define your own voice folder
VOICE_NAME = 'max2'
PRESET = ''

# Ask the user which preset he'd like to use
print('Which preset would you like to use?')
print('(1) ultra_fast')
print('(2) fast')
print('(3) standard (default)')
print('(4) high_quality')
print('(5) custom')

preset = input('Enter a number: ')
if preset == '1':
    PRESET = 'ultra_fast'
    print('Using ultra fast preset.')
elif preset == '2':
    PRESET = 'fast'
    print('Using fast preset.')
elif preset == '3':
    PRESET = 'standard'
    print('Using standard preset.')
elif preset == '4':
    PRESET = 'high_quality'
    print('Using high quality preset.')
elif preset == '5':
    PRESET = 'custom'
else:
    print('Invalid input. Using standard preset.')

# Ask the user what he'd like to say
text = input('What would you like to say? ')

start_time = time.time()

named_tuple = time.localtime()  # get struct_time
time_string = time.strftime("%Y%m%d-%H%M%S", named_tuple)

# Generate with your own voice
voice_samples, conditioning_latents = load_voice(VOICE_NAME)
gen = tts.tts_with_preset(text, voice_samples=voice_samples, conditioning_latents=conditioning_latents,
                          preset=PRESET)
duration = int(time.time() - start_time)
torchaudio.save(os.path.join("results/", f'generated-{VOICE_NAME}-{PRESET}-{time_string}-dura{duration}s.wav'), gen.squeeze(0).cpu(), 24000)
IPython.display.Audio(os.path.join("results/", f'generated-{VOICE_NAME}-{PRESET}-{time_string}-dura{duration}s.wav'))
print("--- %s seconds ---" % duration)
