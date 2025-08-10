# Copyright (c) 2024 Ye Liu. Licensed under the BSD-3-Clause license.
from .language_model.phi3 import MeCoPhi3Config, MeCoPhi3ForCausalLM
from .language_model.qwen2 import MeCoQwen2Config, MeCoQwen2ForCausalLM

MECO_MODELS = {
    'phi3': (MeCoPhi3Config, MeCoPhi3ForCausalLM),
    'qwen2': (MeCoQwen2Config, MeCoQwen2ForCausalLM)
}
