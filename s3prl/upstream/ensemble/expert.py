from typing import Dict, List, Union

import torch.nn as nn
from torch import Tensor, no_grad, tensor
from torch.nn.utils.rnn import pad_sequence

from transformers import (
    Wav2Vec2Config, Wav2Vec2Model, Wav2Vec2Processor,
    HubertConfig, HubertModel,
    WavLMConfig, WavLMModel, 
    WhisperConfig, WhisperModel, WhisperProcessor,
)
from speechbrain.inference.classifiers import EncoderClassifier


def attach_xvector_hooks(xvec_model, layer_ids):
    acts, handles = [], []
    em = getattr(xvec_model.mods, "embedding_model", None)
    if em is None:
        raise RuntimeError("unexpected x-vector model structure: embedding_model not found")

    def _hook(module, inp, out):
        x = out[0] if isinstance(out, (tuple, list)) else out
        acts.append(x.detach().cpu())

    for lid in layer_ids:
        mod = getattr(em, "blocks", None)[lid*3]
        if mod is None:
            raise RuntimeError(f"x-vector layer not found: blocks.{lid*3}")
        handles.append(mod.register_forward_hook(_hook))
    return acts, handles

class UpstreamExpert(nn.Module):
    def __init__(self, ckpt: str = None, model_config: str = None, **kwargs):
        """
        Args:
            ckpt:
                The checkpoint path for loading your pretrained weights.
                Can be assigned by the -k option in run_downstream.py

            model_config:
                The config path for constructing your model.
                Might not needed if you also save that in your checkpoint file.
                Can be assigned by the -g option in run_downstream.py
        """
        super().__init__()
        self.name = "[Example UpstreamExpert]"

        print(
            f"{self.name} - You can use model_config to construct your customized model: {model_config}"
        )
        print(f"{self.name} - You can use ckpt to load your pretrained weights: {ckpt}")
        print(
            f"{self.name} - If you store the pretrained weights and model config in a single file, "
            "you can just choose one argument (ckpt or model_config) to pass. It's up to you!"
        )

        config = Wav2Vec2Config.from_pretrained("facebook/wav2vec2-base")
        self.wav2vec = Wav2Vec2Model(config)
        self.wav2vec.eval()
        self.wav2vec_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")

        config = HubertConfig.from_pretrained("facebook/hubert-base-ls960")
        self.hubert = HubertModel(config)
        self.hubert.eval()
        self.hubert_processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-base-ls960")

        config = WavLMConfig.from_pretrained("microsoft/wavlm-base")
        self.wavlm = WavLMModel(config)
        self.wavlm.eval()
        self.wavlm_processor = Wav2Vec2Processor.from_pretrained("microsoft/wavlm-base")

        config = WhisperConfig.from_pretrained("openai/whisper-small")
        self.whisper = WhisperModel(config)
        self.whisper.eval()
        self.whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-small")

        self.xvector = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
        self.xvector.eval()

    @staticmethod
    def xvector_processor(wavs_list):
        wav_lens = tensor([len(wave) for wave in wavs_list])
        wavs_tensor = pad_sequence(wavs_list, batch_first=True)
        return {
            "wavs": wavs_tensor,
            "wav_lens": wav_lens,
            "normalize": True
        }

    def get_downsample_rates(self, key: str) -> int:
        """
        Since we do not do any downsampling in this example upstream
        All keys' corresponding representations have downsample rate of 1
        """
        return 320

    def forward(self, wavs_list: List[Tensor]) -> Dict[str, Union[Tensor, List[Tensor]]]:
        """
        When the returning Dict contains the List with more than one Tensor,
        those Tensors should be in the same shape to train a weighted-sum on them.
        """

        hidden_states = []
        with no_grad():
            for model, processor in [
                (self.wav2vec, self.wav2vec_processor),
                (self.hubert, self.hubert_processor),
                (self.wavlm, self.wavlm_processor),
                (self.whisper, self.whisper_processor),
            ]:
                inputs = processor(wavs_list, sampling_rate=16000, return_tensors="pt", padding=True)
                outputs = model(**inputs, output_hidden_states=True)
                hidden_states.extend(outputs.hidden_states[[6,9,11]])

        acts, handles = attach_xvector_hooks(self.xvector, layer_ids=[1,2,3])
        with no_grad():
            xvector_inputs = self.xvector_processor(wavs_list)
            _ = self.xvector.encode_batch(**xvector_inputs)
        for h in handles: h.remove()
        hidden_states.extend([act[::2] for act in acts])

        ssl_len = hidden_states[0].size(1)

        hidden_states = [state[:, :ssl_len] for state in hidden_states]

        return {"hidden_states": hidden_states}
