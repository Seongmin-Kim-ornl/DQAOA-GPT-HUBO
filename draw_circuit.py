"""
draw_circuit.py — Visualise a GPT-predicted ADAPT-QAOA circuit.

Generates (or loads) a test HUBO, runs GPT inference to get the predicted
ADAPT operator sequence, then produces two outputs:
  1. ASCII circuit dump via cudaq.draw()
  2. High-level schematic via matplotlib (saved to PNG)

Usage:
    python draw_circuit.py \\
        --ckpt nanoGPT/out-hubo_save_dir/ckpt_33000.pt \\
        --data_dir nanoGPT/data/hubo_save_dir \\
        --n_nodes 6 --num_gpt_samples 5 \\
        --output circuit_diagram.png

    # From formulated HUBO:
    python draw_circuit.py --ckpt ... --n_nodes 6 \\
        --from_formulated --hubo_size 100 --seed 42

    # Save ASCII text to file too:
    python draw_circuit.py --ckpt ... --n_nodes 6 --ascii_out circuit.txt
"""

import os
import sys
import argparse
import contextlib
import io
import pickle
import re

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cudaq

from qaoa_gpt_src.hubo_hamiltonian import (
    generate_random_hubo,
    evaluate_hubo_energy,
    brute_force_hubo,
    hubo_to_spin_ham,
)
from qaoa_gpt_src.hamiltonian_graph import term_coefficients, term_words
from qaoa_gpt_src.adapt_qaoa_pool import all_pool
from utils.solve_HUBO_GPT import _get_hubo_gpt_model
from qaoa_gpt_src.util_hubo import solve_hubo_circuits_with_bitstrings, circ_sanity_check


# ---------------------------------------------------------------------------
# Pool label helpers  (matches all_pool = single_x + mixer + double order)
# ---------------------------------------------------------------------------

def _pair_from_idx(flat_idx, n):
    k = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            if k == flat_idx:
                return i, j
            k += 1
    return 0, 1


def _pool_label(idx, n_nodes):
    """Human-readable label for a pool operator index (0-based, all_pool order)."""
    n = n_nodes
    # single X_i
    if idx < n:
        return f"X_{idx}"
    idx -= n
    # global mixer ΣX
    if idx == 0:
        return "ΣX"
    idx -= 1
    # double ops: XX, YY, YZ, ZY for each pair i<j
    kind_map = ["XX", "YY", "YZ", "ZY"]
    pair_idx = idx // 4
    kind_idx = idx % 4
    i, j = _pair_from_idx(pair_idx, n)
    return f"{kind_map[kind_idx]}_{i}{j}"


def _decode_circuit(q_circuit, n_nodes):
    """Return list of (op_label, beta, gamma) per ADAPT-QAOA layer.

    Token format: [sep_str, op_idx (1-based int), gamma (float), beta (float)] x layers
    """
    layers = []
    for j in range(0, len(q_circuit), 4):
        op_idx = q_circuit[j + 1] - 1   # 1-indexed → 0-indexed
        gamma  = float(q_circuit[j + 2])
        beta   = float(q_circuit[j + 3])
        label  = _pool_label(op_idx, n_nodes)
        layers.append((label, beta, gamma))
    return layers


# ---------------------------------------------------------------------------
# Gate resource counting
# ---------------------------------------------------------------------------

def _pauli_weight(word):
    s  = word if isinstance(word, str) else ''.join(word)
    k  = sum(1 for c in s if c != 'I')
    nx = s.count('X')
    ny = s.count('Y')
    return k, nx, ny


def _count_circuit_resources(cost_raw_words, mixer_pool_strings, n_nodes):
    """
    Estimate gate counts for ADAPT-QAOA.
    Each ADAPT layer contributes: cost terms + mixer operator terms.

    exp_pauli(weight-k Pauli):
      1q: 2*(nX+nY) basis changes + 1 Rz
      2q: 2*(k-1) CNOTs
    """
    n1q = n_nodes   # initial H⊗n
    n2q = 0
    depth = 1       # H layer
    n_adapt = len(mixer_pool_strings)

    def _tally(word):
        nonlocal n1q, n2q, depth
        k, nx, ny = _pauli_weight(word)
        if k == 0:
            return
        n1q   += 2 * (nx + ny) + 1
        cnots  = 2 * (k - 1) if k >= 2 else 0
        n2q   += cnots
        depth += 2 * (nx + ny) + cnots + 1

    for _ in range(n_adapt):
        for w in cost_raw_words:
            _tally(w)

    for ops in mixer_pool_strings:
        for w in ops:
            _tally(w)

    return n1q, n2q, depth, n_adapt


# ---------------------------------------------------------------------------
# cudaq.draw  (ASCII)
# ---------------------------------------------------------------------------

def draw_cudaq_ascii(T, q_circuit, n_nodes):
    """Return cudaq ASCII string for the ADAPT-QAOA kernel."""
    spin_ham   = hubo_to_spin_ham(T)
    ham_coeffs = term_coefficients(spin_ham)
    ham_words  = term_words(spin_ham, n_nodes)

    pools = all_pool(n_nodes)

    op_indices, gamma_coef, beta_coef = [], [], []
    for j in range(0, len(q_circuit), 4):
        op_indices.append(q_circuit[j + 1] - 1)
        gamma_coef.append(float(q_circuit[j + 2]))
        beta_coef.append(float(q_circuit[j + 3]))

    mixer_pool = []
    for idx in op_indices:
        op = pools[idx]
        mixer_pool.append(
            [cudaq.pauli_word(term.get_pauli_word(n_nodes)) for term in op])

    num_layer = len(op_indices)

    @cudaq.kernel
    def kernel_qaoa(qubits_num: int,
                    h_words:    list[cudaq.pauli_word],
                    h_coeffs:   list[complex],
                    mp:         list[list[cudaq.pauli_word]],
                    gamma:      list[float],
                    beta:       list[float],
                    nl:         int):
        qubits = cudaq.qvector(qubits_num)
        h(qubits)
        for p in range(nl):
            for i in range(len(h_coeffs)):
                exp_pauli(gamma[p] * h_coeffs[i].real, qubits, h_words[i])
            for word in mp[p]:
                exp_pauli(beta[p], qubits, word)

    return cudaq.draw(
        kernel_qaoa,
        n_nodes, ham_words, ham_coeffs,
        mixer_pool, gamma_coef, beta_coef, num_layer,
    )


# ---------------------------------------------------------------------------
# Matplotlib schematic
# ---------------------------------------------------------------------------

def draw_schematic(T, layers, n_nodes, energy_gpt, energy_ref,
                   output_path, title_extra="",
                   n1q=None, n2q=None, circuit_depth=None):
    """
    Block diagram of the ADAPT-QAOA circuit.

    Each ADAPT layer is two side-by-side sub-boxes, both spanning all n qubits:
      left  (orange): cost unitary U_C(γ_p)
      right (color) : mixer operator exp(iβ P)
    """
    n_layers = len(layers)

    # Column widths (data units): each ADAPT layer = two half-width sub-blocks
    col_h_w    = 0.8
    col_adpt_w = 1.6   # wide enough for two sub-blocks side by side
    col_meas_w = 0.8

    n_cols_adapt = max(n_layers, 1)
    total_w = col_h_w + n_cols_adapt * col_adpt_w + col_meas_w + 0.4

    row_h    = 0.7
    margin_l = 0.7
    margin_r = 0.4
    n_info_lines = (3
                    + (1 if energy_ref is not None else 0)
                    + (1 if n1q is not None else 0)
                    + (1 if n2q is not None else 0)
                    + (1 if circuit_depth is not None else 0))
    margin_t = max(1.8, 0.32 * n_info_lines + 0.6)
    margin_b = 0.5

    fig_w = margin_l + total_w + margin_r + 0.5
    fig_h = margin_b + n_nodes * row_h + margin_t + 0.2

    fig, ax = plt.subplots(figsize=(max(fig_w, 8), max(fig_h, 4)))
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.axis('off')

    BG_COL    = '#FAFAFA'
    WIRE_COL  = '#555555'
    H_COL     = '#4A90D9'
    COST_COL  = '#E67E22'
    ADAPT_COLS = [
        '#27AE60', '#8E44AD', '#16A085', '#C0392B', '#2980B9',
        '#D35400', '#1ABC9C', '#9B59B6', '#F39C12', '#2ECC71',
    ]
    MEAS_COL  = '#7F8C8D'

    fig.patch.set_facecolor(BG_COL)
    ax.set_facecolor(BG_COL)

    # Title
    title = f"ADAPT-QAOA Circuit  (n={n_nodes},  layers={n_layers})"
    if title_extra:
        title += f"\n{title_extra}"
    ax.set_title(title, fontsize=11, fontweight='bold', pad=8,
                 color='#2C3E50', loc='left', x=margin_l / fig_w)

    def qy(q):
        return margin_b + (n_nodes - 1 - q) * row_h + row_h / 2

    y_bot = qy(n_nodes - 1) - 0.22
    y_top = qy(0) + 0.22
    y_ctr = (y_bot + y_top) / 2
    box_h = y_top - y_bot

    # Wires
    wire_x_end = margin_l + total_w + 0.1
    for q in range(n_nodes):
        y = qy(q)
        ax.plot([margin_l - 0.1, wire_x_end], [y, y],
                color=WIRE_COL, linewidth=1.2, zorder=1)
        ax.text(margin_l - 0.15, y, f"q{q}",
                ha='right', va='center', fontsize=8, color='#2C3E50')

    # H block
    x0 = margin_l
    for q in range(n_nodes):
        y = qy(q)
        rect = FancyBboxPatch(
            (x0, y - 0.22), col_h_w - 0.1, 0.44,
            boxstyle="round,pad=0.03",
            facecolor=H_COL, edgecolor='white', linewidth=1.2, zorder=2)
        ax.add_patch(rect)
        ax.text(x0 + (col_h_w - 0.1) / 2, y, "H",
                ha='center', va='center', fontsize=9,
                fontweight='bold', color='white', zorder=3)

    # ADAPT-QAOA layers: left sub-block = U_C (full height), right = mixer (full height)
    x1 = x0 + col_h_w
    sub_w = col_adpt_w / 2 - 0.08   # width of each sub-block

    for li, (op_label, beta, gamma) in enumerate(layers):
        col = ADAPT_COLS[li % len(ADAPT_COLS)]
        xl = x1 + li * col_adpt_w

        # Left sub-block: cost unitary (full height, all qubits)
        x_cost = xl + 0.04
        cost_rect = FancyBboxPatch(
            (x_cost, y_bot), sub_w, box_h,
            boxstyle="round,pad=0.03",
            facecolor=COST_COL, edgecolor='white', linewidth=1.2,
            alpha=0.92, zorder=2)
        ax.add_patch(cost_rect)
        ax.text(x_cost + sub_w / 2, y_ctr + 0.08,
                "U_C(γ)", ha='center', va='center',
                fontsize=7.5, fontweight='bold', color='white', zorder=3)
        ax.text(x_cost + sub_w / 2, y_ctr - 0.12,
                f"γ={gamma:.3f}", ha='center', va='center',
                fontsize=6, color='#FFE0B2', zorder=3)

        # Right sub-block: mixer operator (full height, all qubits)
        x_mix = xl + col_adpt_w / 2 + 0.04
        mix_rect = FancyBboxPatch(
            (x_mix, y_bot), sub_w, box_h,
            boxstyle="round,pad=0.03",
            facecolor=col, edgecolor='white', linewidth=1.2,
            alpha=0.88, zorder=2)
        ax.add_patch(mix_rect)
        ax.text(x_mix + sub_w / 2, y_ctr + 0.08,
                op_label, ha='center', va='center',
                fontsize=7.5, fontweight='bold', color='white', zorder=3)
        ax.text(x_mix + sub_w / 2, y_ctr - 0.12,
                f"β={beta:.3f}", ha='center', va='center',
                fontsize=6, color='#E8F5E9', zorder=3)

        # Layer index below
        ax.text(xl + col_adpt_w / 2, y_bot - 0.25, f"L{li + 1}",
                ha='center', va='top', fontsize=7, color='#555', zorder=3)

    if n_layers == 0:
        xl = x1
        ax.text(xl + col_adpt_w / 2, y_ctr,
                "(no ADAPT\nlayers)", ha='center', va='center',
                fontsize=8, color='#999', zorder=3)

    # Measurement block
    x2 = x1 + n_cols_adapt * col_adpt_w
    for q in range(n_nodes):
        y = qy(q)
        m_rect = FancyBboxPatch(
            (x2 + 0.05, y - 0.22), col_meas_w - 0.1, 0.44,
            boxstyle="round,pad=0.03",
            facecolor=MEAS_COL, edgecolor='white', linewidth=1.0, zorder=2)
        ax.add_patch(m_rect)
        ax.text(x2 + col_meas_w / 2, y, "M",
                ha='center', va='center', fontsize=9,
                fontweight='bold', color='white', zorder=3)

    # Info annotations
    info_x = margin_l
    info_y = margin_b + n_nodes * row_h + 0.35
    info_lines = [f"GPT energy : {energy_gpt:.6f}"]
    if energy_ref is not None:
        ar = energy_gpt / energy_ref if energy_ref < -1e-8 else float('nan')
        info_lines.append(f"Ref energy : {energy_ref:.6f}")
        info_lines.append(f"Approx ratio: {ar:.4f}")
    if n1q is not None:
        info_lines.append(f"1q gates   : {n1q}")
    if n2q is not None:
        info_lines.append(f"2q gates   : {n2q}  (CNOTs)")
    if circuit_depth is not None:
        info_lines.append(f"Depth      : {circuit_depth}  (serial est.)")
    for li2, line in enumerate(info_lines):
        ax.text(info_x, info_y + li2 * 0.28, line,
                ha='left', va='bottom', fontsize=8.5,
                color='#2C3E50', fontfamily='monospace')

    # Legend
    legend_patches = [
        mpatches.Patch(color=H_COL,        label='Hadamard (H⊗n)'),
        mpatches.Patch(color=COST_COL,     label='Cost unitary U_C(γ)'),
        mpatches.Patch(color=ADAPT_COLS[0], label='Mixer operator exp(iβP)'),
        mpatches.Patch(color=MEAS_COL,     label='Measurement'),
    ]
    ax.legend(handles=legend_patches, loc='upper right',
              fontsize=7.5, framealpha=0.9,
              bbox_to_anchor=(1.0 - margin_r / fig_w, 1.0),
              bbox_transform=ax.transAxes)

    plt.tight_layout(pad=0.3)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor=BG_COL)
    plt.close()
    print(f"  Schematic saved → {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Draw GPT-predicted ADAPT-QAOA circuit",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ckpt",          type=str, required=True)
    p.add_argument("--data_dir",      type=str, default=None)
    p.add_argument("--config_file",   type=str, default=None)
    p.add_argument("--n_nodes",       type=int, default=6)
    p.add_argument("--seed",          type=int, default=None,
                   help="Random seed for HUBO sampling (default: random)")
    p.add_argument("--density",       type=float, default=0.5)
    p.add_argument("--weight_range",  type=float, default=1.0)
    p.add_argument("--from_formulated", action="store_true")
    p.add_argument("--hubo_size",     type=int, default=None,
                   help="Required with --from_formulated")
    p.add_argument("--formulated_dir", type=str, default=None,
                   help="Directory with {hubo_size}_QUBO.txt and {hubo_size}_k_third.txt")
    p.add_argument("--hubo_npy",      type=str, default=None,
                   help="Path to .npy file of a pre-built full HUBO tensor")
    p.add_argument("--gamma",         type=float, default=None,
                   help="Override per-layer gamma from circuit tokens with a fixed value "
                        "(useful when model outputs gamma=0)")
    p.add_argument("--device",        type=str, default="cuda",
                   choices=["cuda", "cpu", "mps"])
    p.add_argument("--num_gpt_samples", type=int, default=5)
    p.add_argument("--temperature",   type=float, default=0.8)
    p.add_argument("--pool_type",     type=str, default="all_pool",
                   choices=["all_pool", "qaoa_mixer", "qaoa_single_x", "qaoa_double_ops"])
    p.add_argument("--num_shots",     type=int, default=1024)
    p.add_argument("--output",        type=str, default="qaoa_circuit.png")
    p.add_argument("--ascii_out",     type=str, default=None,
                   help="Also save cudaq.draw() ASCII to this file")
    p.add_argument("--no_brute_force", action="store_true")
    return p.parse_args()


def main():
    args     = parse_args()
    ckpt     = args.ckpt
    data_dir = args.data_dir or os.path.dirname(os.path.abspath(ckpt))
    config_file = args.config_file or os.path.join(data_dir, 'train_adapt_gpt_config.py')

    for path, label in [(ckpt, "checkpoint"), (data_dir, "data_dir"),
                        (config_file, "config_file")]:
        if not os.path.exists(path):
            print(f"ERROR: {label} not found: {path}")
            sys.exit(1)

    # ---- Generate / sample test HUBO ----------------------------------------
    import random
    seed = args.seed if args.seed is not None else random.randint(0, 2**31)
    print(f"Using seed: {seed}")

    if args.from_formulated:
        if args.hubo_size is None:
            print("ERROR: --hubo_size required with --from_formulated")
            sys.exit(1)
        from subHUBO_gen import formulate_HUBO, make_sub_HUBO
        T_full = formulate_HUBO(args.hubo_size, data_dir=args.formulated_dir)
        random.seed(seed)
        T, _ = make_sub_HUBO(T_full, args.n_nodes)
        print(f"HUBO: sub-HUBO (n={args.n_nodes}) from formulated HUBO (size={args.hubo_size})")
    elif args.hubo_npy:
        from subHUBO_gen import make_sub_HUBO
        T_full = np.load(args.hubo_npy)
        random.seed(seed)
        T, _ = make_sub_HUBO(T_full, args.n_nodes)
        print(f"HUBO: sub-HUBO (n={args.n_nodes}) from {args.hubo_npy}")
    else:
        rng = np.random.default_rng(seed)
        T = generate_random_hubo(
            args.n_nodes, density=args.density,
            weight_range=args.weight_range,
            seed=int(rng.integers(0, 2**31)),
        )
        print(f"HUBO: random n_nodes={args.n_nodes}")

    # ---- Reference energy ---------------------------------------------------
    energy_ref = None
    if not args.no_brute_force and args.n_nodes <= 20:
        _, energy_ref, _ = brute_force_hubo(T)
        print(f"Brute-force minimum energy: {energy_ref:.6f}")

    # ---- GPT inference -------------------------------------------------------
    _rd = re.search(r"token_seq_round\s*=\s*['\"].*_d(\d+)['\"]",
                    open(config_file).read())
    rounding_digits = int(_rd.group(1)) if _rd else 2
    T_rounded = np.round(T.copy(), rounding_digits)

    _meta_path = os.path.join(data_dir, 'meta.pkl')
    with open(_meta_path, 'rb') as f:
        meta = pickle.load(f)
    stoi = meta['stoi']
    vocab_floats = np.array(sorted(
        [k for k in stoi if isinstance(k, float) and not isinstance(k, bool)]))
    if len(vocab_floats) > 0:
        vf_min, vf_max = vocab_floats[0], vocab_floats[-1]
        def _snap(val):
            if val == 0.0:
                return 0.0
            clipped = float(np.clip(val, vf_min, vf_max))
            return float(vocab_floats[np.argmin(np.abs(vocab_floats - clipped))])
        T_rounded = np.vectorize(_snap)(T_rounded)

    print(f"Running GPT inference  (num_gpt_samples={args.num_gpt_samples})...")

    @contextlib.contextmanager
    def _silence():
        with contextlib.redirect_stdout(io.StringIO()), \
             contextlib.redirect_stderr(io.StringIO()):
            yield

    from tqdm import tqdm
    tqdm.disable = True

    with _silence():
        gpt_model = _get_hubo_gpt_model(ckpt, data_dir, config_file, args.device)
        circ_df = gpt_model.generate_circ_from_sub_hubos(
            {'sub_hubo_0': T_rounded},
            n_samples_per_batch=50,
            num_samples=args.num_gpt_samples,
            max_new_tokens=200,
            temperature=args.temperature,
            top_k=200,
            rounding_digits=rounding_digits,
        )

    # ---- Collect valid circuits, pick best by energy ------------------------
    all_circs = []
    for _, row in circ_df.iterrows():
        for c in row.get('q_circuits', []):
            if circ_sanity_check(c):
                all_circs.append(c)
        ac = row.get('adapt_circuit', [])
        if ac and circ_sanity_check(ac):
            all_circs.append(ac)

    if not all_circs:
        print("ERROR: No valid circuits produced by GPT.")
        sys.exit(1)

    print(f"Valid circuits from GPT: {len(all_circs)}")

    results_df = solve_hubo_circuits_with_bitstrings(
        qaoa_gpt_circ_df=circ_df,
        T_dict={'sub_hubo_0': T_rounded},
        n_nodes=args.n_nodes,
        pool_type=args.pool_type,
        num_shots=args.num_shots,
        verbose=False,
    )

    best_energy_gpt = float('inf')
    best_bs         = '0' * args.n_nodes
    best_circ       = all_circs[0]

    for _, row in results_df.iterrows():
        candidates = list(row.get('adapt_gpt_bitstrings', []))
        adapt_bs   = row.get('ADAPT_bitstring', None)
        if adapt_bs:
            candidates.append(adapt_bs)
        circs_row = [c for c in row.get('q_circuits', []) if circ_sanity_check(c)]
        ac_row    = row.get('adapt_circuit', [])
        if ac_row and circ_sanity_check(ac_row):
            circs_row.append(ac_row)
        for bs_cand in candidates:
            if len(bs_cand) != args.n_nodes:
                continue
            x_cand = np.array([int(b) for b in bs_cand], dtype=np.uint8)
            e_cand  = evaluate_hubo_energy(T, x_cand)
            if e_cand < best_energy_gpt:
                best_energy_gpt = e_cand
                best_bs         = bs_cand
                if circs_row:
                    best_circ = circs_row[0]

    energy_gpt = best_energy_gpt
    layers     = _decode_circuit(best_circ, args.n_nodes)

    # Optional gamma override (model sometimes outputs gamma=0 when OOD)
    if args.gamma is not None:
        layers = [(lbl, beta, args.gamma) for lbl, beta, _ in layers]
        print(f"  [gamma override] All per-layer γ set to {args.gamma}")

    print(f"\nSelected circuit (best energy across all GPT samples):")
    print(f"  GPT energy   : {energy_gpt:.6f}  bitstring={best_bs}")
    if energy_ref is not None:
        ar = energy_gpt / energy_ref if energy_ref < -1e-8 else float('nan')
        print(f"  Ref energy   : {energy_ref:.6f}")
        print(f"  Approx ratio : {ar:.4f}")
    print(f"\nADAPT-QAOA layers ({len(layers)}):")
    for li, (lbl, beta, gamma) in enumerate(layers):
        print(f"  L{li + 1:2d}  op={lbl:12s}  β={beta:.6f}  γ={gamma:.6f}")

    # ---- cudaq ASCII --------------------------------------------------------
    print("\nGenerating cudaq ASCII circuit...")
    try:
        ascii_str = draw_cudaq_ascii(T, best_circ, args.n_nodes)
        print(ascii_str)
        if args.ascii_out:
            with open(args.ascii_out, 'w') as af:
                af.write(ascii_str)
            print(f"  ASCII saved → {args.ascii_out}")
    except Exception as exc:
        print(f"  [WARN] cudaq.draw failed: {exc}")
        ascii_str = ""

    # ---- Gate resource summary ----------------------------------------------
    _spin_ham   = hubo_to_spin_ham(T)
    _raw_words  = [w for c, w in zip(term_coefficients(_spin_ham),
                                     term_words(_spin_ham, args.n_nodes))
                   if set(w) != {'I'}]
    pools_rc    = all_pool(args.n_nodes)
    op_indices_rc = [best_circ[j + 1] - 1 for j in range(0, len(best_circ), 4)]
    mixer_pool_strings = [
        [term.get_pauli_word(args.n_nodes) for term in pools_rc[idx]]
        for idx in op_indices_rc
    ]
    n1q, n2q, depth, n_adapt = _count_circuit_resources(
        _raw_words, mixer_pool_strings, args.n_nodes)
    print(f"\nCircuit resources (estimated):")
    print(f"  1-qubit gates : {n1q}")
    print(f"  2-qubit gates : {n2q}  (CNOTs)")
    print(f"  Circuit depth : {depth}  (serial)")
    print(f"  ADAPT layers  : {n_adapt}")

    # ---- Matplotlib schematic -----------------------------------------------
    print("\nGenerating schematic PNG...")
    title_extra = f"bitstring={best_bs}  E={energy_gpt:.4f}"
    if energy_ref is not None:
        ar = energy_gpt / energy_ref if energy_ref < -1e-8 else float('nan')
        title_extra += f"  ref={energy_ref:.4f}  AR={ar:.4f}"

    draw_schematic(T, layers, args.n_nodes,
                   energy_gpt, energy_ref,
                   output_path=args.output,
                   title_extra=title_extra,
                   n1q=n1q, n2q=n2q, circuit_depth=depth)

    print("\nDone.")


if __name__ == "__main__":
    main()
