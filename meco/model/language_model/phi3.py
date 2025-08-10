# Copyright (c) 2024 Ye Liu. Licensed under the BSD-3-Clause license.
import torch
import numpy as np
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, Phi3Config, Phi3ForCausalLM, Phi3Model
from transformers.models.phi3.modeling_phi3 import Phi3DecoderLayer, Phi3Attention
from meco.model.meco_arch import MeCoMetaForCausalLM, MeCoMetaModel


class MeCoPhi3Config(Phi3Config):
    model_type = 'MeCo_phi3'


class MeCoPhi3Model(MeCoMetaModel, Phi3Model):
    config_class = MeCoPhi3Config

class FrameHeadPhi3(Phi3DecoderLayer):
    def __init__(self, config: Phi3Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = Phi3Attention(config, layer_idx=layer_idx)

class MeCoPhi3ForCausalLM(MeCoMetaForCausalLM, Phi3ForCausalLM):
    config_class = MeCoPhi3Config

    def __init__(self, config):
        super().__init__(config)
        self.model = MeCoPhi3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.localization:
            hidden_size = self.config.hidden_size
            self.vid_head = nn.Sequential(
                    nn.LayerNorm(hidden_size), nn.Linear(hidden_size, hidden_size // 2), nn.GELU(),
                    nn.Linear(hidden_size // 2, hidden_size))

            if config.use_mha:
                self.frm_head = FrameHeadPhi3(config, layer_idx=-1)
            else: 
                self.frm_head = nn.Sequential(
                    nn.LayerNorm(hidden_size), nn.Linear(hidden_size, hidden_size // 2), nn.GELU(),
                    nn.Linear(hidden_size // 2, hidden_size))

            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.post_init()

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

AutoConfig.register('MeCo_phi3', MeCoPhi3Config)
AutoModelForCausalLM.register(MeCoPhi3Config, MeCoPhi3ForCausalLM)
