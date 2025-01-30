"""Extraction of HuBERT features from audio signals

    :class:`~shennong.audio.Audio` ---> HubertProcessor \
    ---> :class:`~shennong.features.Features`

Examples
--------

>>> from shennong.audio import Audio
>>> from shennong.processor.hubert import HubertProcessor
>>> audio = Audio.load('./test/data/test.wav')
>>> processor = HubertProcessor(model_path='/home/exp/mhubert-147')

Compute the HuBERT features. the output is an
instance of :class:`~shennong.features.Features`:

>>> hubert = processor.process(audio, layer=3)
>>> type(hubert)
<class 'shennong.features.Features'>

References
----------

.. [HuBERT] https://arxiv.org/abs/2106.07447

"""

import torch
import fairseq

import numpy as np

from shennong import Features
from shennong.processor.base import FeaturesProcessor
from transformers import HubertForCTC

class HubertProcessor(FeaturesProcessor):
    """HuBERT features from a pre-trained neural network

    Parameters
    ----------
    model_path : The path to the pre-trained HuBERT model

    layer : The layer to extract features from

    Raises
    ------
    RuntimeError
        If the model path does not point to a HuBERT model 
        that can be loaded with either fairseq or huggingface

    ValueError
        If the selected layer does not exist in the given model.
    """

    _SEED = 3939

    def __init__(self, model_path="", layer=""):
        super().__init__()
        torch.manual_seed(self._SEED)
        np.random.seed(self._SEED)

        self.model_path = model_path
        try:
            self.model = fairseq.checkpoint_utils.load_model_ensemble_and_task([self.model_path])[0][0]
            self._model_type = 'fairseq'
        except:
            try:
                self.model = HubertForCTC.from_pretrained(self.model_path)
                self._model_type = 'huggingface'
            except:
                raise RuntimeError(f"The model at {self.model_path} cannot be loaded. Make sure that this is a fairseq model or huggingface model directory.")
        
        self._check_layer(layer, self.model)
        self.layer = layer

    @property
    def name(self):
        return 'hubert'

    @property
    def model_path(self):
        """The path to the pretrained HuBERT model"""
        return self._model_path

    @model_path.setter
    def model_path(self, value):
        self._model_path = str(value)

    @property
    def layer(self):
        """The layer to extract features from"""
        return self._layer

    @layer.setter
    def layer(self, value):
        self._layer = int(value)

    @property
    def ndims(self):
        """The dimension of extracted frames

        Cannot be tuned because the underlying neural networks are
        trained with this parameter.

        """
        return 768
    
    @property
    def sample_rate(self):
        """Processing sample frequency in Hertz

        Cannot be tuned because the underlying neural networks are
        trained with this parameter.

        """
        return 16000
    
    @property
    def frame_length(self):
        """The length of extracted frames (in seconds)

        Cannot be tuned because the underlying neural networks are
        trained with this parameter.

        """
        return 0.02

    @property
    def frame_shift(self):
        """The time shift between two consecutive frames (in seconds)

        Cannot be tuned because the underlying neural networks are
        trained with this parameter.

        """
        return 0.02
    
    def _check_layer(self, value, model):
        if self._model_type == 'fairseq':
            layer_num = len(model.encoder.layers)
        elif self._model_type == 'huggingface':
            layer_num = model.config.num_hidden_layers

        if value not in range(layer_num + 1):
            raise ValueError(f"Layer {value} does not exist in this model")
        elif not value:
            raise ValueError("No layers selected")

    def process(self, signal):
        """Computes HuBERT features with the specified options

        Use a pre-trained neural network to extract HuBERT
        features. Features have a frame shift of 20 ms and frame
        length of 20 ms.

        Parameters
        ----------
        signal : Audio, shape = [nsamples, 1]
            The input audio signal to compute the features on, must be
            mono. The signal is up/down-sampled to 16 kHz during
            processing.

        Returns
        -------
        features : Features, shape = [nframes, 768]
            The computed HuBERT features will have as many rows as
            there are frames (depends on the `signal` duration, expect
            about 50 frames per second), each frame with 768
            dimensions.

        Raises
        ------
        ValueError
            If the input `signal` has more than one channel (i.e. is
            not mono). If `sample_rate` != `signal.sample_rate`.
        """

        self.model.eval()

        # ensure the signal is correct
        if signal.nchannels != 1:
            raise ValueError(
                'signal must have one dimension, but it has {}'
                .format(signal.nchannels))

        # force resampling to 16 kHz and 32 bit floats
        need_resample = (
            signal.sample_rate != 16000 or
            signal.dtype is not np.dtype(np.float32))

        if need_resample:
            self.log.debug(
                'resampling audio from %dHz@%db to %dHz@%db',
                signal.sample_rate, signal.dtype.itemsize * 8, 16000, 32)
            signal = signal.resample(16000).astype(np.float32)

        signal = torch.unsqueeze(torch.from_numpy(signal.data), 0)

        if self._model_type == 'fairseq':
            out_dict = self.model(signal, features_only=True, mask=False, output_layer=self.layer)
            data = out_dict["features"][0].squeeze(1).detach().numpy()
        elif self._model_type == 'huggingface':
            out_dict = self.model(signal, output_hidden_states=True)
            data = out_dict["hidden_states"][self.layer][0].squeeze(1).detach().numpy()
        
        del out_dict

        # compute the timestamps for each output frame
        times = np.vstack((
            np.arange(data.shape[0]) * self.frame_shift,
            np.arange(data.shape[0]) * self.frame_shift + self.frame_length)).T

        return Features(
            data, times, properties=self.get_properties())