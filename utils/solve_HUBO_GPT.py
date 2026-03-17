import os
import sys
import contextlib
import io

import numpy as np
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
nanogpt_dir = os.path.join(project_root, 'nanoGPT')
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if nanogpt_dir not in sys.path:
    sys.path.insert(0, nanogpt_dir)

from qaoa_gpt_src.model_interface_hubo import HUBO_GPT
from qaoa_gpt_src.util_hubo import solve_hubo_circuits_with_bitstrings

_hubo_gpt_model_cache = {}

def _get_hubo_gpt_model(model_ckpt, data_dir, config_file, device):
    key = (model_ckpt, data_dir, config_file, device)
    if key not in _hubo_gpt_model_cache:
        _hubo_gpt_model_cache[key] = HUBO_GPT(
            model_ckpt=model_ckpt,
            data_dir=data_dir,
            config_file=config_file,
            device=device,
        )
    return _hubo_gpt_model_cache[key]

def solve_sub_HUBO_GPT(sub_HUBO,
                        model_ckpt=None,
                        data_dir=None,
                        config_file=None,
                        device='cuda',
                        num_shots=1024,
                        pool_type='all_pool',
                        verbose=False):
    sub_HUBO = np.round(sub_HUBO, 2)
    n_nodes = sub_HUBO.shape[0]

    if model_ckpt is None:
        gpt_output_dir = 'out-hubo_save_dir'
        ckpt_fname = 'ckpt_hubo.pt'
        model_ckpt = os.path.join(nanogpt_dir, gpt_output_dir, ckpt_fname)

    if data_dir is None:
        data_dir = os.path.dirname(os.path.abspath(model_ckpt))

    if config_file is None:
        config_file = os.path.join(data_dir, 'train_adapt_gpt_config.py')

    import re
    with open(config_file, 'r') as _cf:
        _cfg_text = _cf.read()
    _match = re.search(r'^\s*n_nodes\s*=\s*(\d+)', _cfg_text, re.MULTILINE)
    if _match:
        _cfg_n_nodes = int(_match.group(1))
        assert n_nodes == _cfg_n_nodes, \
            f"--sub_n {n_nodes} doesn't match model's n_nodes={_cfg_n_nodes} " \
            f"(checkpoint: {model_ckpt})"

    @contextlib.contextmanager
    def _silence():
        with contextlib.redirect_stdout(io.StringIO()), \
             contextlib.redirect_stderr(io.StringIO()):
            yield

    tqdm.disable = True

    with _silence():
        hubo_gpt_obj = _get_hubo_gpt_model(model_ckpt, data_dir, config_file, device)

    import pickle
    _meta_path = os.path.join(data_dir, 'meta.pkl')
    with open(_meta_path, 'rb') as _f:
        _meta = pickle.load(_f)
    _stoi = _meta['stoi']
    _vocab_nums = np.array(sorted([k for k in _stoi if isinstance(k, (int, float))]))
    if len(_vocab_nums) > 0:
        def _snap(val):
            if val == 0.0:
                return 0.0
            return float(_vocab_nums[np.argmin(np.abs(_vocab_nums - val))])
        _snapped = np.vectorize(_snap)(sub_HUBO)
        sub_HUBO = _snapped

    with _silence():
        qaoa_gpt_circ_df = hubo_gpt_obj.generate_circ_from_sub_hubos(
            {'sub_hubo_0': sub_HUBO},
            n_samples_per_batch=50,
            num_samples=1,
            max_new_tokens=150,
            temperature=0.1,
            top_k=200,
        )

    T_dict = {'sub_hubo_0': sub_HUBO}

    results_df = solve_hubo_circuits_with_bitstrings(
        qaoa_gpt_circ_df=qaoa_gpt_circ_df,
        T_dict=T_dict,
        n_nodes=n_nodes,
        pool_type=pool_type,
        num_shots=num_shots,
        verbose=verbose,
    )

    bitstring = '0' * n_nodes
    for _, row in results_df.iterrows():
        bitstring = row.get('ADAPT_bitstring', '0' * n_nodes)

    binary_vector = np.array([int(b) for b in bitstring], dtype=np.uint8)
    return binary_vector
