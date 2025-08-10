# Copyright (c) 2024 Ye Liu. Licensed under the BSD-3-Clause license.
import torch
import numpy as np
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, Qwen2Config, Qwen2ForCausalLM, Qwen2Model

from meco.model.meco_arch import MeCoMetaForCausalLM, MeCoMetaModel
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2Attention

class MeCoQwen2Config(Qwen2Config):
    model_type = 'meco_qwen2'

class MeCoQwen2Model(MeCoMetaModel, Qwen2Model):
    config_class = MeCoQwen2Config

class FrameHeadQwen2(Qwen2DecoderLayer):
    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = Qwen2Attention(config, layer_idx=layer_idx)

class MeCoQwen2ForCausalLM(MeCoMetaForCausalLM, Qwen2ForCausalLM):
    config_class = MeCoQwen2Config

    def __init__(self, config):
        super().__init__(config)
        self.model = MeCoQwen2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.localization:
            hidden_size = self.config.hidden_size
            self.vid_head = nn.Sequential(
                    nn.LayerNorm(hidden_size), nn.Linear(hidden_size, hidden_size // 2), nn.GELU(),
                    nn.Linear(hidden_size // 2, hidden_size))

            if config.use_mha:
                self.frm_head = FrameHeadQwen2(config, layer_idx=-1)
            else: 
                nn.Sequential(
                    nn.LayerNorm(hidden_size), nn.Linear(hidden_size, hidden_size // 2), nn.GELU(),
                    nn.Linear(hidden_size // 2, hidden_size))

            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)) 

        self.post_init()

AutoConfig.register('meco_qwen2', MeCoQwen2Config)
AutoModelForCausalLM.register(MeCoQwen2Config, MeCoQwen2ForCausalLM)
