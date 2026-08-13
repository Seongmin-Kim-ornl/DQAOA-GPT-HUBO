"""
utils/solve_HUBO_GPT_gpt_only.py — solve_sub_HUBO_GPT variant that samples
ONLY the GPT-generated candidate circuits, never the extra adapt_circuit
slot that utils/solve_HUBO_GPT.py's solve_sub_HUBO_GPT always samples too.

Kept as a separate new file rather than editing utils/solve_HUBO_GPT.py or
qaoa_gpt_src/util_hubo.py — don't modify widely-shared pipeline files.

Why this matters: for any inference-only sub-HUBO (random/formulated/
decomposed — anything that isn't a training-data row with a real recorded
ADAPT-QAOA solution attached), adapt_circuit is always [] (empty). The
original solve_sub_HUBO_GPT still samples it anyway — a real
cudaq.sample() call on a 0-layer, H-only circuit whose result is pure
uniform-superposition noise. For plain timing that's just wasted compute;
for an inference/AR test specifically it's worse than wasted: that noise
result is still added as a scored CANDIDATE in the best-energy comparison,
so a lucky/unlucky random draw can occasionally influence which bitstring
gets reported as best, for no principled reason. This variant never
generates, samples, or scores it — with --num_gpt_samples 1, it executes
EXACTLY 1 circuit per problem, not 2, and the reported result depends only
on what GPT actually proposed.

Reuses (does not duplicate) the real circuit-building/execution code —
qaoa_gpt_src.util_hubo.sample_ansatz_bitstring_only_hubo — completely
unmodified; only the outer loop that decides which circuits to sample
differs from the original solve_hubo_circuits_with_bitstrings.
"""

import os
import sys
import contextlib
import io
import pickle
import re

import numpy as np
import pandas as pd
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
nanogpt_dir = os.path.join(project_root, 'nanoGPT')
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if nanogpt_dir not in sys.path:
    sys.path.insert(0, nanogpt_dir)

from qaoa_gpt_src.model_interface_hubo import HUBO_GPT
from qaoa_gpt_src.util_hubo import sample_ansatz_bitstring_only_hubo
from qaoa_gpt_src.hubo_hamiltonian import evaluate_hubo_energy

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


def _solve_hubo_circuits_gpt_only(qaoa_gpt_circ_df, T_dict, n_nodes, pool_type,
                                    num_shots, verbose):
    """Same as qaoa_gpt_src.util_hubo.solve_hubo_circuits_with_bitstrings,
    except it NEVER samples rec['adapt_circuit'] — only rec['q_circuits']."""
    records = qaoa_gpt_circ_df.to_dict('records')
    for rec in records:
        graph_prefix = rec['graph_prefix']
        T = T_dict.get(graph_prefix)
        q_circuits = rec.get('q_circuits', [])
        bitstrings = []
        if T is not None:
            for circ in q_circuits:
                bitstring = '0' * n_nodes
                try:
                    bitstring = sample_ansatz_bitstring_only_hubo(
                        T, circ, n_nodes, pool_type,
                        num_shots=num_shots, verbose=verbose)
                except Exception:
                    pass
                bitstrings.append(bitstring)
        rec['adapt_gpt_bitstrings'] = bitstrings
    return pd.DataFrame(records)


def solve_sub_HUBO_GPT_gpt_only(sub_HUBO,
                                 model_ckpt=None,
                                 data_dir=None,
                                 config_file=None,
                                 device='cuda',
                                 num_shots=1024,
                                 pool_type='all_pool',
                                 num_gpt_samples=10,
                                 temperature=0.8,
                                 verbose=False):
    """Same interface/behavior as utils.solve_HUBO_GPT.solve_sub_HUBO_GPT,
    except only the num_gpt_samples GPT-generated circuits are ever
    executed — the extra adapt_circuit slot (always empty/NULL for
    inference-only sub-HUBOs) is never sampled or scored. With
    num_gpt_samples=1, this runs exactly 1 circuit, not 2."""
    n_nodes = sub_HUBO.shape[0]
    T_original = sub_HUBO.copy()

    if model_ckpt is None:
        gpt_output_dir = 'out-hubo_save_dir'
        ckpt_fname = 'ckpt_hubo.pt'
        model_ckpt = os.path.join(nanogpt_dir, gpt_output_dir, ckpt_fname)

    if data_dir is None:
        data_dir = os.path.dirname(os.path.abspath(model_ckpt))

    if config_file is None:
        config_file = os.path.join(data_dir, 'train_adapt_gpt_config.py')

    with open(config_file, 'r') as _cf:
        _cfg_text = _cf.read()
    _match = re.search(r'^\s*n_nodes\s*=\s*(\d+)', _cfg_text, re.MULTILINE)
    if _match:
        _cfg_n_nodes = int(_match.group(1))
        assert n_nodes == _cfg_n_nodes, \
            f"--sub_n {n_nodes} doesn't match model's n_nodes={_cfg_n_nodes} " \
            f"(checkpoint: {model_ckpt})"
    _rd_match = re.search(r"token_seq_round\s*=\s*['\"].*_d(\d+)['\"]", _cfg_text)
    _rounding_digits = int(_rd_match.group(1)) if _rd_match else 2

    sub_HUBO = np.round(sub_HUBO, _rounding_digits)

    @contextlib.contextmanager
    def _silence():
        with contextlib.redirect_stdout(io.StringIO()), \
             contextlib.redirect_stderr(io.StringIO()):
            yield

    tqdm.disable = True

    with _silence():
        hubo_gpt_obj = _get_hubo_gpt_model(model_ckpt, data_dir, config_file, device)

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
            num_samples=num_gpt_samples,
            max_new_tokens=150,
            temperature=temperature,
            top_k=200,
            rounding_digits=_rounding_digits,
        )

    T_dict = {'sub_hubo_0': sub_HUBO}

    results_df = _solve_hubo_circuits_gpt_only(
        qaoa_gpt_circ_df=qaoa_gpt_circ_df,
        T_dict=T_dict,
        n_nodes=n_nodes,
        pool_type=pool_type,
        num_shots=num_shots,
        verbose=verbose,
    )

    best_bitstring = '0' * n_nodes
    best_energy = float('inf')

    for _, row in results_df.iterrows():
        for bs in row.get('adapt_gpt_bitstrings', []):
            if len(bs) != n_nodes:
                continue
            x = np.array([int(b) for b in bs], dtype=np.uint8)
            e = evaluate_hubo_energy(T_original, x)
            if e < best_energy:
                best_energy = e
                best_bitstring = bs

    binary_vector = np.array([int(b) for b in best_bitstring], dtype=np.uint8)
    return binary_vector
