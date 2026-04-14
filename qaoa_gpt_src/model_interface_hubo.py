import pickle
from contextlib import nullcontext
import torch
import pandas as pd
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

from nanoGPT.model_pad_gemb import GPTConfig as GPTConfig_gemb
from nanoGPT.model_pad_gemb import GPT as GPT_gemb
from nanoGPT.model_pad import GPTConfig as GPTConfig_nogemb
from nanoGPT.model_pad import GPT as GPT_nogemb

from qaoa_gpt_src.util_hubo import (
    prepare_model_input_hubo,
    extract_graph_hubo,
    circ_sanity_check,
)
from qaoa_gpt_src.util import generate_circ_from_df

dtype_str_to_torch_dict = {
    "float32": torch.float32,
    "float": torch.float32,
    "float16": torch.float16,
    "half": torch.float16,
    "bfloat16": torch.bfloat16,
    "float64": torch.float64,
    "double": torch.float64,
}

class HUBO_GPT:

    def __init__(self,
                 model_ckpt,
                 config_file,
                 data_dir,
                 device,
                 n_nodes='infer',
                 temp_folder='adapt_gpt_temp_data'):

        config_fpath = Path(config_file)
        assert config_fpath.is_file(), f"Config not found: {config_fpath}"

        print(f"Loading HUBO_GPT config from: {config_fpath}")
        config_vars = {}
        with open(config_fpath) as f:
            exec(f.read(), config_vars)

        self.pool_type = config_vars['pool_type']
        self.use_graph_emb = config_vars['use_graph_emb']

        if 'n_nodes' not in config_vars:
            if n_nodes == 'infer':
                raise AttributeError(
                    "n_nodes not in config. Pass n_nodes=<N> to HUBO_GPT constructor.")
            self.n_nodes = int(n_nodes)
        else:
            self.n_nodes = int(config_vars['n_nodes'])

        self.data_dir = Path(data_dir)
        self.model_ckpt = Path(model_ckpt)
        self.temp_folder = Path(temp_folder)
        self.device = device
        self.seed = 1337

        if self.device == 'cuda':
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                self.dtype = 'bfloat16'
            else:
                self.dtype = 'float16'
        else:
            self.dtype = 'float16'

        self.compile = False

        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        self.device_type = 'cuda' if 'cuda' in self.device else 'cpu'
        ptdtype = dtype_str_to_torch_dict[self.dtype]
        self.ctx = (nullcontext() if self.device_type == 'cpu'
                    else torch.amp.autocast(device_type=self.device_type, dtype=ptdtype))

        self.meta = pd.read_pickle(f'{data_dir}/meta.pkl')

        self.gptconfig = GPTConfig_gemb if self.use_graph_emb else GPTConfig_nogemb
        self.gpt_cls = GPT_gemb if self.use_graph_emb else GPT_nogemb

        self.model = self._load_model(self.model_ckpt)

    def _load_model(self, model_fpath):
        checkpoint = torch.load(Path(model_fpath), map_location=self.device)
        gptconf = self.gptconfig(**checkpoint['model_args'])
        model = self.gpt_cls(gptconf)
        state_dict = checkpoint['model']
        for k in list(state_dict.keys()):
            if k.startswith('_orig_mod.'):
                state_dict[k[len('_orig_mod.'):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        model.eval()
        model.to(self.device)
        return model

    def generate_circ_from_sub_hubos(self,
                                      sub_hubos_container,
                                      n_samples_per_batch=50,
                                      num_samples=1,
                                      max_new_tokens=150,
                                      temperature=0.1,
                                      top_k=200,
                                      rounding_digits=2):
        if isinstance(sub_hubos_container, list):
            sub_hubos_dict = {f'sub_hubo_{i}': T
                              for i, T in enumerate(sub_hubos_container)}
        elif isinstance(sub_hubos_container, dict):
            sub_hubos_dict = sub_hubos_container
        else:
            raise ValueError("sub_hubos_container must be a list or dict.")

        hubo_nx_df, feather_par_emb, emb_graph_id_to_idx = prepare_model_input_hubo(
            sub_hubos_dict,
            n_nodes=self.n_nodes,
            rounding_digits=rounding_digits,
        )

        emb_dtype = (self.dtype if self.device != 'cpu' else 'float')

        gc_df = generate_circ_from_df(
            hubo_nx_df,
            graph_emb_np=feather_par_emb,
            emb_graph_id_to_idx_dict=emb_graph_id_to_idx,
            meta=self.meta,
            model=self.model,
            device=self.device,
            ctx=self.ctx,
            n_samples_per_batch=n_samples_per_batch,
            num_samples=num_samples,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            token_seq_col='token_seq_round_d2',
            normalize_weights_flag=False,
            emb_dtype=dtype_str_to_torch_dict[emb_dtype],
        )

        return gc_df
