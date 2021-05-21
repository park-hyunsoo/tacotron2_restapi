import os
import json
import numpy as np
import torch

import sys
sys.path.append('waveglow/')

from hparams import defaults
from model import Tacotron2
from layers import TacotronSTFT, STFT
from audio_processing import griffin_lim
from train import load_model
from text import text_to_sequence
from waveglow.glow import WaveGlow
import soundfile as sf

_CHECKPOINT_DIR = '/home/web/checkpoint'

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

class TacotronHandler:
    def __init__(self):
        super().__init__()
        self.tacotron_model = None
        self.waveglow = None

    def _load_tacotron2(self, checkpoint_path):
        hparams = Struct(**defaults)
        hparams.sampling_rate = 22050

        model = load_model(hparams)
        model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
        model.cuda().eval()
        self.tacotron_model = model

    def _load_waveglow(self, checkpoint_path):
        waveglow = torch.hub.load('nvidia/DeepLearningExamples:torchhub', 'nvidia_waveglow')
        waveglow = waveglow.remove_weightnorm(waveglow)
        waveglow = waveglow.to('cuda')
        waveglow.eval()
        self.waveglow = waveglow

    def initialize(self):
        if not torch.cuda.is_available():
            raise RuntimeError("This model is not supported on CPU machines.")

        self._load_tacotron2(checkpoint_path=os.path.join(_CHECKPOINT_DIR,'checkpoint_138500'))
        self._load_waveglow(checkpoint_path=os.path.join(_CHECKPOINT_DIR,'waveglow.pt'))


    def preprocess(self, text):
        sequence = np.array(text_to_sequence(text, ['korean_cleaners']))[None, :]
        sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()
        return sequence

    def inference(self, sequence):
        with torch.no_grad():
            mel_outputs, mel_outputs_postnet, _, alignments = self.tacotron_model.inference(sequence)
            audio = self.waveglow.infer(mel_outputs_postnet, sigma=0.666)
        audio_numpy = audio[0].data.cpu().numpy()
        return audio_numpy

    def postprocess(self, audio):
        sf.write('tts_output.wav', audio, 22050)
        return 'tts_output.wav'
