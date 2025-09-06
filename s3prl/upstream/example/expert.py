from typing import Dict, List, Union

import torch.nn as nn
from torch import Tensor, load, no_grad, zeros_like
from torch.nn.utils.rnn import pad_sequence

from transformers import Wav2Vec2Config, Wav2Vec2Model

from prior.nn.model.cnn1d import CNN1dKernel


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
        self.model = Wav2Vec2Model(config)
        checkpoint = load(ckpt)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        prototype = CNN1dKernel(3, 5)
        prototype.load_state_dict(checkpoint['prototype_state_dict'])

        self.postnet = prototype.export_network(768, 768, 768)

    def get_downsample_rates(self, key: str) -> int:
        """
        Since we do not do any downsampling in this example upstream
        All keys' corresponding representations have downsample rate of 1
        """
        return 320

    def forward(self, wavs: List[Tensor]) -> Dict[str, Union[Tensor, List[Tensor]]]:
        """
        When the returning Dict contains the List with more than one Tensor,
        those Tensors should be in the same shape to train a weighted-sum on them.
        """
        wavs = [(wave - wave.mean()) / (wave.std() + 1e-7) for wave in wavs]
        lens = [len(wave) for wave in wavs]
        wavs = pad_sequence(wavs, batch_first=True)
        # wavs: (batch_size, max_len)
        mask = zeros_like(wavs).long()
        for i, l in enumerate(lens):
            mask[i, :l] = 1

        with no_grad():
            hidden_state = self.model.forward(wavs, attention_mask=mask).last_hidden_state

        hidden_state = self.postnet(hidden_state.transpose(-2, -1)).transpose(-2, -1)

        return {
            "hidden_states": [hidden_state],
        }
