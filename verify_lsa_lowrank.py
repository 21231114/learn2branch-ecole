"""
verify_lsa_lowrank.py — 验证 AdaptiveSlicing (LSA) 所处理的节点特征矩阵 z_var_0 是否低秩。

思路:
    1. 加载训练好的 OptiFlow 模型
    2. 从 SC 测试集采样若干实例
    3. 前向传播至 GraphInitialization，得到 z_var_0 [N_var, D]
    4. 对 z_var_0 做 SVD，分析奇异值谱
    5. 输出有效秩、能量集中度等指标，并绘制奇异值衰减曲线

用法:
    python verify_lsa_lowrank.py
"""

import os
import sys
import pathlib
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = ''  # CPU suff够

import torch
import gzip
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ======================================================================
# 加载模型
# ======================================================================

def load_model(model_path, config_dir):
    """加载 OptiFlow 模型。"""
    import json
    from model.graph_init import GraphInitialization
    from model.adaptive_slicing import AdaptiveSlicing
    from model.latent_evolution import LatentTrajectoryEvolution
    from model.deslicing_decoder import DeslicingDecoder

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    spec = __import__('03_train_optiflow', fromlist=['OptiFlowModel'])
    OptiFlowModel = spec.OptiFlowModel

    cfg_path = config_dir / 'config.json'
    if cfg_path.exists():
        with open(cfg_path) as f:
            config = json.load(f)
    else:
        config = {
            'cons_nfeats': 5, 'var_nfeats': 23,
            'emb_size': 64, 'n_slices': 64, 'n_evolve_steps': 3,
        }

    model = OptiFlowModel(
        cons_nfeats=config.get('cons_nfeats', 5),
        var_nfeats=config.get('var_nfeats', 23),
        emb_size=config.get('emb_size', 64),
        n_slices=config.get('n_slices', 64),
        n_evolve_steps=config.get('n_evolve_steps', 3),
        dropout=0.0,
    )

    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
    model.eval()
    print(f"模型已加载: {model_path}")
    print(f"  emb_size={config.get('emb_size', 64)}, "
          f"n_slices={config.get('n_slices', 64)}")
    return model, config


# ======================================================================
# 加载样本数据
# ======================================================================

def load_sample(sample_path):
    """加载单个 .pkl 样本，返回 tensor。"""
    with gzip.open(sample_path, 'rb') as f:
        sample = pickle.load(f)

    cons_feats, (edge_idx, edge_vals), var_feats = sample['observation']
    cons_feats = torch.FloatTensor(np.asarray(cons_feats, dtype=np.float32))
    edge_idx = torch.LongTensor(np.asarray(edge_idx, dtype=np.int64))
    edge_vals = torch.FloatTensor(
        np.asarray(edge_vals, dtype=np.float32)).unsqueeze(-1)
    var_feats = torch.FloatTensor(np.asarray(var_feats, dtype=np.float32))

    return cons_feats, edge_idx, edge_vals, var_feats


# ======================================================================
# 秩分析
# ======================================================================

def analyze_rank(matrix_np, label="z_var_0"):
    """
    对矩阵做 SVD，返回分析结果字典。

    Parameters
    ----------
    matrix_np : ndarray [N, D]
    label : str

    Returns
    -------
    dict with keys:
        shape, svd_values, rank_full, effective_rank_90, effective_rank_95,
        effective_rank_99, stable_rank, top1_energy, top5_energy, top10_energy
    """
    N, D = matrix_np.shape
    # full SVD
    U, S, Vt = np.linalg.svd(matrix_np, full_matrices=False)

    total_energy = np.sum(S ** 2)
    cumulative_energy = np.cumsum(S ** 2) / total_energy

    # 数值秩 (阈值 1e-5 × 最大奇异值)
    tol = 1e-5 * S[0] if S[0] > 0 else 1e-10
    rank_full = int(np.sum(S > tol))

    # 有效秩 (达到 X% 能量所需的奇异值数)
    eff_rank_90 = int(np.searchsorted(cumulative_energy, 0.90)) + 1
    eff_rank_95 = int(np.searchsorted(cumulative_energy, 0.95)) + 1
    eff_rank_99 = int(np.searchsorted(cumulative_energy, 0.99)) + 1

    # 稳定秩 (stable rank = ||A||_F^2 / ||A||_2^2)
    stable_rank = total_energy / (S[0] ** 2) if S[0] > 0 else 0

    # 前 k 个奇异值的能量占比
    top1 = cumulative_energy[0] if len(S) >= 1 else 0
    top5 = cumulative_energy[min(4, len(S) - 1)]
    top10 = cumulative_energy[min(9, len(S) - 1)]

    return {
        'label': label,
        'shape': (N, D),
        'max_possible_rank': min(N, D),
        'svd_values': S,
        'cumulative_energy': cumulative_energy,
        'rank_numerical': rank_full,
        'eff_rank_90': eff_rank_90,
        'eff_rank_95': eff_rank_95,
        'eff_rank_99': eff_rank_99,
        'stable_rank': stable_rank,
        'top1_energy': top1,
        'top5_energy': top5,
        'top10_energy': top10,
    }


def print_analysis(info):
    """打印单个矩阵的分析结果。"""
    print(f"\n  --- {info['label']} ---")
    print(f"  形状: {info['shape'][0]} x {info['shape'][1]}")
    print(f"  最大可能秩: {info['max_possible_rank']}")
    print(f"  数值秩 (tol=1e-5*sigma_1): {info['rank_numerical']}")
    print(f"  有效秩 (90% 能量): {info['eff_rank_90']}")
    print(f"  有效秩 (95% 能量): {info['eff_rank_95']}")
    print(f"  有效秩 (99% 能量): {info['eff_rank_99']}")
    print(f"  稳定秩 (||A||_F^2 / ||A||_2^2): {info['stable_rank']:.2f}")
    print(f"  前 1 个奇异值能量占比: {info['top1_energy']:.4f}")
    print(f"  前 5 个奇异值能量占比: {info['top5_energy']:.4f}")
    print(f"  前10 个奇异值能量占比: {info['top10_energy']:.4f}")

    S = info['svd_values']
    print(f"  奇异值谱 (前10): {', '.join(f'{s:.4f}' for s in S[:10])}")
    if len(S) > 10:
        print(f"  奇异值谱 (后5):  {', '.join(f'{s:.6f}' for s in S[-5:])}")


# ======================================================================
# 绘图
# ======================================================================

def plot_results(all_results, output_dir):
    """
    绘制两张图:
        1. 奇异值衰减曲线 (log scale)
        2. 累积能量曲线
    """
    os.makedirs(output_dir, exist_ok=True)

    n_samples = len(all_results)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # -- 图1: 奇异值衰减 --
    ax = axes[0]
    for i, info in enumerate(all_results):
        S = info['svd_values']
        # 归一化 (除以最大奇异值)
        S_norm = S / S[0] if S[0] > 0 else S
        ax.semilogy(range(1, len(S_norm) + 1), S_norm,
                     marker='o', markersize=3, label=info['label'], alpha=0.8)
    ax.set_xlabel('Singular Value Index')
    ax.set_ylabel('Normalized Singular Value (log scale)')
    ax.set_title('Singular Value Decay of z_var_0\n(Input to AdaptiveSlicing / LSA)')
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, alpha=0.3)

    # -- 图2: 累积能量 --
    ax = axes[1]
    for i, info in enumerate(all_results):
        ce = info['cumulative_energy']
        ax.plot(range(1, len(ce) + 1), ce,
                marker='s', markersize=3, label=info['label'], alpha=0.8)

    ax.axhline(y=0.90, color='gray', linestyle='--', alpha=0.5, label='90% energy')
    ax.axhline(y=0.95, color='gray', linestyle='-.', alpha=0.5, label='95% energy')
    ax.axhline(y=0.99, color='gray', linestyle=':', alpha=0.5, label='99% energy')
    ax.set_xlabel('Number of Singular Values')
    ax.set_ylabel('Cumulative Energy Fraction')
    ax.set_title('Cumulative Energy of z_var_0\n(How many singular values capture X% energy?)')
    ax.legend(fontsize=7, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    out_path = os.path.join(output_dir, 'lsa_lowrank_analysis.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\n图已保存: {out_path}")

    # -- 图3: 各样本有效秩对比 (bar chart) --
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = [info['label'] for info in all_results]
    max_ranks = [info['max_possible_rank'] for info in all_results]
    ranks_90 = [info['eff_rank_90'] for info in all_results]
    ranks_95 = [info['eff_rank_95'] for info in all_results]
    ranks_99 = [info['eff_rank_99'] for info in all_results]
    stable_ranks = [info['stable_rank'] for info in all_results]

    x = np.arange(len(labels))
    width = 0.15

    ax.bar(x - 2 * width, max_ranks, width, label='Max possible rank', color='lightgray')
    ax.bar(x - width, ranks_99, width, label='Eff. rank (99%)', color='#2ca02c')
    ax.bar(x, ranks_95, width, label='Eff. rank (95%)', color='#1f77b4')
    ax.bar(x + width, ranks_90, width, label='Eff. rank (90%)', color='#ff7f0e')
    ax.bar(x + 2 * width, stable_ranks, width, label='Stable rank', color='#d62728')

    ax.set_xlabel('Sample')
    ax.set_ylabel('Rank')
    ax.set_title('Effective Rank of z_var_0 (Input to LSA)\nLower effective rank = stronger low-rank structure')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    out_path2 = os.path.join(output_dir, 'lsa_effective_rank_comparison.png')
    plt.savefig(out_path2, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"图已保存: {out_path2}")


# ======================================================================
# 额外分析: 原始特征矩阵 vs GNN 输出
# ======================================================================

def analyze_raw_vs_gnn(raw_var_feats_np, z_var_0_np, label):
    """对比原始变量特征和 GNN 输出的秩结构。"""
    raw_info = analyze_rank(raw_var_feats_np, label=f"{label} (raw var_feats)")
    gnn_info = analyze_rank(z_var_0_np, label=f"{label} (z_var_0)")

    print(f"\n  ====== 原始特征 vs GNN 输出对比 [{label}] ======")
    print(f"  {'':30s} {'Raw var_feats':>15s} {'z_var_0 (GNN)':>15s}")
    print(f"  {'-'*62}")
    print(f"  {'Shape':30s} {str(raw_info['shape']):>15s} {str(gnn_info['shape']):>15s}")
    print(f"  {'数值秩':30s} {raw_info['rank_numerical']:>15d} {gnn_info['rank_numerical']:>15d}")
    print(f"  {'有效秩 (90%)':30s} {raw_info['eff_rank_90']:>15d} {gnn_info['eff_rank_90']:>15d}")
    print(f"  {'有效秩 (95%)':30s} {raw_info['eff_rank_95']:>15d} {gnn_info['eff_rank_95']:>15d}")
    print(f"  {'有效秩 (99%)':30s} {raw_info['eff_rank_99']:>15d} {gnn_info['eff_rank_99']:>15d}")
    print(f"  {'稳定秩':30s} {raw_info['stable_rank']:>15.2f} {gnn_info['stable_rank']:>15.2f}")
    print(f"  {'前1 SV 能量':30s} {raw_info['top1_energy']:>15.4f} {gnn_info['top1_energy']:>15.4f}")
    print(f"  {'前5 SV 能量':30s} {raw_info['top5_energy']:>15.4f} {gnn_info['top5_energy']:>15.4f}")
    print(f"  {'前10 SV 能量':30s} {raw_info['top10_energy']:>15.4f} {gnn_info['top10_energy']:>15.4f}")

    return raw_info, gnn_info


# ======================================================================
# Main
# ======================================================================

if __name__ == '__main__':
    MODEL_PATH = pathlib.Path(
        '/home/lmh/private/learn2branch-ecole/trained_models/optiflow/SC/2/best_model.pt')
    DATA_DIR = pathlib.Path(
        '/home/lmh/private/learn2branch-ecole/data/samples/SC/test')
    OUTPUT_DIR = pathlib.Path(
        '/home/lmh/private/learn2branch-ecole/lsa_rank_analysis_output')
    N_SAMPLES = 5

    # ---- 加载模型 ----
    model, config = load_model(MODEL_PATH, MODEL_PATH.parent)

    # ---- 采样测试数据 ----
    sample_files = sorted(DATA_DIR.glob('sample_*.pkl'))
    if len(sample_files) == 0:
        print(f"ERROR: {DATA_DIR} 中没有 sample_*.pkl 文件")
        sys.exit(1)

    # 均匀采样
    indices = np.linspace(0, len(sample_files) - 1,
                          min(N_SAMPLES, len(sample_files)), dtype=int)
    selected_files = [sample_files[i] for i in indices]
    print(f"\n已选择 {len(selected_files)} 个样本进行分析:")
    for f in selected_files:
        print(f"  {f.name}")

    # ---- 逐样本分析 ----
    all_z_var_results = []
    all_raw_results = []

    print("\n" + "=" * 70)
    print("开始分析 z_var_0 (GraphInitialization 输出 / AdaptiveSlicing 输入)")
    print("=" * 70)

    for i, fpath in enumerate(selected_files):
        label = fpath.stem  # e.g., "sample_10"
        print(f"\n[{i+1}/{len(selected_files)}] 处理 {fpath.name} ...")

        cons_feats, edge_idx, edge_vals, var_feats = load_sample(fpath)
        print(f"  约束节点: {cons_feats.shape[0]}, 变量节点: {var_feats.shape[0]}, "
              f"边数: {edge_idx.shape[1]}")

        with torch.no_grad():
            # Step 1 only: GraphInitialization → z_var_0
            z_var_0 = model.graph_init(cons_feats, edge_idx, edge_vals, var_feats)

        z_var_0_np = z_var_0.cpu().numpy()
        var_feats_np = var_feats.cpu().numpy()

        # 分析 z_var_0 的秩
        info = analyze_rank(z_var_0_np, label=label)
        print_analysis(info)
        all_z_var_results.append(info)

        # 对比 raw vs GNN
        raw_info, gnn_info = analyze_raw_vs_gnn(var_feats_np, z_var_0_np, label)
        all_raw_results.append(raw_info)

    # ---- 汇总统计 ----
    print("\n" + "=" * 70)
    print("汇总统计")
    print("=" * 70)

    print(f"\n{'Sample':<20s} {'Shape':>12s} {'MaxRank':>8s} {'NumRank':>8s} "
          f"{'Eff90':>6s} {'Eff95':>6s} {'Eff99':>6s} {'StableR':>8s} "
          f"{'Top1%':>7s} {'Top5%':>7s}")
    print("-" * 100)

    for info in all_z_var_results:
        print(f"{info['label']:<20s} "
              f"{str(info['shape']):>12s} "
              f"{info['max_possible_rank']:>8d} "
              f"{info['rank_numerical']:>8d} "
              f"{info['eff_rank_90']:>6d} "
              f"{info['eff_rank_95']:>6d} "
              f"{info['eff_rank_99']:>6d} "
              f"{info['stable_rank']:>8.2f} "
              f"{info['top1_energy']:>7.4f} "
              f"{info['top5_energy']:>7.4f}")

    avg_eff90 = np.mean([r['eff_rank_90'] for r in all_z_var_results])
    avg_eff95 = np.mean([r['eff_rank_95'] for r in all_z_var_results])
    avg_eff99 = np.mean([r['eff_rank_99'] for r in all_z_var_results])
    avg_stable = np.mean([r['stable_rank'] for r in all_z_var_results])
    avg_max = np.mean([r['max_possible_rank'] for r in all_z_var_results])

    print("-" * 100)
    print(f"{'平均':<20s} {'':>12s} {avg_max:>8.0f} {'':>8s} "
          f"{avg_eff90:>6.1f} {avg_eff95:>6.1f} {avg_eff99:>6.1f} "
          f"{avg_stable:>8.2f}")

    # 低秩判断
    print("\n" + "=" * 70)
    print("低秩判断")
    print("=" * 70)
    ratio_90 = avg_eff90 / avg_max
    ratio_95 = avg_eff95 / avg_max
    ratio_99 = avg_eff99 / avg_max
    print(f"  平均有效秩(90%) / 最大可能秩 = {avg_eff90:.1f} / {avg_max:.0f} = {ratio_90:.4f}")
    print(f"  平均有效秩(95%) / 最大可能秩 = {avg_eff95:.1f} / {avg_max:.0f} = {ratio_95:.4f}")
    print(f"  平均有效秩(99%) / 最大可能秩 = {avg_eff99:.1f} / {avg_max:.0f} = {ratio_99:.4f}")
    print(f"  平均稳定秩 / 最大可能秩       = {avg_stable:.2f} / {avg_max:.0f} = {avg_stable/avg_max:.4f}")

    if ratio_95 < 0.5:
        print(f"\n  结论: z_var_0 呈现明显的低秩结构。")
        print(f"         95% 能量仅需 {avg_eff95:.1f} / {avg_max:.0f} 个奇异值 "
              f"(占 {ratio_95:.1%})。")
        print(f"         这为 AdaptiveSlicing 用 K={config.get('n_slices', 64)} 个切片")
        print(f"         进行低秩近似提供了理论支撑。")
    else:
        print(f"\n  结论: z_var_0 的秩相对较满 (95% 能量需 {ratio_95:.1%} 的奇异值)。")
        print(f"         低秩近似的压缩效果可能有限。")

    # ---- 绘图 ----
    plot_results(all_z_var_results, str(OUTPUT_DIR))

    print(f"\n所有输出已保存至: {OUTPUT_DIR}")
