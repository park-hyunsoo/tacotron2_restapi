import numpy as np
import torch

from models import Tacotron2
from hparams import create_hparams
from layers import TacotronSTFT, STFT
from audio_processing import griffin_lim

from scipy.io.wavfile import write
import soundfile as sf

class TacotronHandler(nn.Module):
    def __init__(self):
        super().__init__()
        self.tacotron_model = None
        self.waveglow = None
        self.device = None
        self.denoiser = None

    def _load_model(hparams):
        model = Tacotron2(hparams).cuda()
        if hparams.fp16_run:
            model.decoder.attention_layer.score_mask_value = finfo('float16').min

        if hparams.distributed_run:
            model = apply_gradient_allreduce(model)

        return model

    def _load_tacotron2(self, checkpoint_file):
        tacotron2_checkpoint = torch.load(checkpoint_file)

        hparams = create_hparams()
        hparams.sampling_rate = 22050

        self.tacotron_model = self._load_model(hparams)
        self.tacotron_model.load_state_dict(tacotron2_checkpoint['state_dict'])
        self.tacotron_model.to(self.device)
        self.tacotron_model.eval().half()


    def _load_waveglow(self, checkpoint_file):
        self.waveglow = torch.load(waveglow_path)['model']
        self.waveglow.cuda().eval().half()
        for k in self.waveglow.convinv:
            k.float()

    def initialize(self):
        if not torch.cuda.is_available():
            raise RuntimeError("This model is not supported on CPU machines.")
        self.device = torch.device('cuda')

        self._load_tacotron2(checkpoint_file='tacotron2.pt')
        self._load_waveglow(checkpoint_file='waveglow_weights.pt')


    def preprocess(self, text):
        sequence = np.array(text_to_sequence(text, ['korean_cleaners']))[None, :]
        sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()
        return sequence

    def inference(self, sequence):
        mel_outputs, mel_outputs_postnet, _, alignments = self.tacotron_model.inference(sequence)
        with torch.no_grad():
            audio = self.waveglow.infer(mel_output_postnet, sigma=0.666)
        return audio

    def postprocess(self, audio):
        sf.write('tts_output.wav', audio, '22050')
        return 'API/audio/output_name'


handler = TacotronHandler()
handler.initialize()
sequence = handler.preprocess('안녕하세요')
audio = handler.inference(sequence)
handler.postprocess(audio)

audio_denoised = handler.denoise(audio)
handler.postprocess(audio_denoised)

