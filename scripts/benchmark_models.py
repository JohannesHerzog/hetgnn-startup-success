"""
Benchmark model architectures: parameter count, training time, inference time.
Outputs a LaTeX-ready table for the thesis.

Models benchmarked (S/M/L = hidden 32/64/128):
  - MLP        (hidden=64, no graph)
  - GCN        (hidden=64, num_layers=1/2)
  - GraphSAGE  (hidden=64, num_layers=1/2)
  - HAN-S/M/L  (heterogeneous attention, num_layers=1/2)
  - SeHGNN-S/M/L (transformer metapath fusion, num_hops=1/2)

All models use same graph data.
"""
import sys
import time
import copy
import json
import torch
import numpy as np
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ml.utils import load_config
from src.ml.preprocessing import perform_preprocessing
from src.ml.train import Trainer
from torch_geometric import seed_everything


def count_parameters(model):
    """Count trainable and total parameters."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def format_params(n):
    """Format parameter count as human-readable string."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    elif n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def time_inference(model, graph_data, device, n_runs=5):
    """Time inference (forward pass) averaged over n_runs."""
    model.eval()
    data = graph_data.to(device)

    # Warmup
    with torch.no_grad():
        _ = model(data.x_dict, data.edge_index_dict)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    times = []
    for _ in range(n_runs):
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(data.x_dict, data.edge_index_dict)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    return np.mean(times), np.std(times)


def benchmark_model(config, model_name, model_overrides, graph_data, device, epochs=10, config_overrides=None):
    """Run a full benchmark for one model configuration."""
    cfg = copy.deepcopy(config)
    cfg['train']['model'] = model_name
    cfg['train']['device'] = device.type
    cfg['train']['epochs'] = epochs
    cfg['wandb']['enabled'] = False
    cfg['analysis']['enable_homophily_analysis'] = False
    cfg['analysis']['enable_downstream_analysis'] = False
    cfg['visualize']['enabled'] = False
    cfg['visualize']['feature_visualization'] = False

    # Apply config-level overrides (e.g. max_metapaths)
    if config_overrides:
        for key_path, value in config_overrides.items():
            keys = key_path.split('.')
            d = cfg
            for k in keys[:-1]:
                d = d[k]
            d[keys[-1]] = value

    # Apply model-specific overrides
    if model_name in cfg['models']:
        cfg['models'][model_name].update(model_overrides)

    # Disable retrieval head for clean benchmark
    if model_name == 'SeHGNN':
        cfg['models']['SeHGNN']['use_retrieval_head'] = False

    # Reset GPU memory tracking
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()

    # Create trainer (this builds the model)
    trainer = Trainer(graph_data, cfg)
    model = trainer.model

    # Get actual metapath count for SeHGNN
    num_metapaths = len(model.metapaths) if hasattr(model, 'metapaths') else None

    # Time training
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t_train_start = time.perf_counter()
    trainer.train()
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t_train_total = time.perf_counter() - t_train_start
    t_per_epoch = t_train_total / epochs

    # Count parameters (after training so LazyModules are initialized)
    trainable, total = count_parameters(model)

    # Peak GPU memory after training
    peak_mem_train = torch.cuda.max_memory_allocated() / 1e9 if device.type == 'cuda' else None

    # Time inference
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    inf_mean, inf_std = time_inference(model, graph_data, device)
    peak_mem_inf = torch.cuda.max_memory_allocated() / 1e9 if device.type == 'cuda' else None

    # Load best model and run test evaluation
    if trainer.best_model_dict is not None:
        model.load_state_dict(trainer.best_model_dict)
    target_mode = cfg['data_processing']['target_mode']
    test_scalar, test_metrics = trainer.evaluator.evaluate(
        graph_data=trainer.data,
        mode="test",
        target_mode=target_mode,
    )

    return {
        'model': model_name,
        'trainable_params': trainable,
        'total_params': total,
        'train_time_total': t_train_total,
        'train_time_per_epoch': t_per_epoch,
        'peak_mem_train_gb': peak_mem_train,
        'peak_mem_inf_gb': peak_mem_inf,
        'num_metapaths': num_metapaths,
        'inference_mean': inf_mean,
        'inference_std': inf_std,
        'epochs': epochs,
        'test_metrics': test_metrics,
    }


def main():
    seed_everything(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load config and preprocess data once
    config = load_config("config.yaml")
    config['train']['device'] = device.type

    # Enable metapath discovery for SeHGNN benchmark
    config['data_processing']['add_metapaths'] = True

    print("\n=== Loading data ===")
    graph_data, node_names = perform_preprocessing(
        startups_filename="startup_nodes.csv",
        investors_filename="investor_nodes.csv",
        founders_filename="founder_nodes.csv",
        cities_filename="city_nodes.csv",
        university_filename="university_nodes.csv",
        sectors_filename="sector_nodes.csv",
        startup_investor_filename="startup_investor_edges.csv",
        startup_city_filename="startup_city_edges.csv",
        startup_founder_filename="startup_founder_edges.csv",
        startup_sector_filename="startup_sector_edges.csv",
        founder_university_filename="founder_university_edges.csv",
        investor_city_filename="investor_city_edges.csv",
        investor_sector_filename="investor_sector_edges.csv",
        university_city_filename="university_city_edges.csv",
        founder_investor_employment_filename="founder_investor_employment_edges.csv",
        founder_coworking_filename="founder_coworking_edges.csv",
        founder_investor_identity_filename="founder_investor_identity_edges.csv",
        startup_descriptively_similar_filename="startup_descriptively_similar_edges.csv",
        founder_descriptively_similar_filename="founder_descriptively_similar_edges.csv",
        founder_co_study_filename="founder_co_study_edges.csv",
        founder_board_filename="founder_board_edges.csv",
        founder_startup_director_filename="founder_startup_director_edges.csv",
        founder_investor_director_filename="founder_investor_director_edges.csv",
        config=config,
    )

    EPOCHS = 100

    # Size tiers: S=32, M=64, L=128
    S, M, L = 32, 64, 128

    # Shared SeHGNN params (from sweep rj010kwp, rounded)
    SEHGNN_BASE = {
        'dropout': 0.15, 'input_drop': 0.35, 'att_drop': 0.3,
        'activation_type': 'leaky_relu', 'transformer_activation': 'relu',
        'use_residual': True, 'use_self_loop': True, 'num_hops': 1,
    }

    models = [
        # ============================================================
        # PART 1: SIZE COMPARISON (num_layers=2 for all)
        # ============================================================

        # --- Non-graph baseline ---
        ("MLP", "MLP", {'hidden_channels': M, 'num_layers': 2}),

        # --- Homogeneous GNNs (fixed at M=64) ---
        ("GCN", "GCN", {'hidden_channels': M, 'num_layers': 2}),
        ("GraphSAGE", "SageGNN", {'hidden_channels': M, 'num_layers': 2}),

        # --- HAN S/M/L ---
        ("HAN-S", "HAN", {
            'hidden_channels': S, 'num_layers': 2, 'heads': 2,
            'dropout': 0.24, 'activation_type': 'prelu',
        }),
        ("HAN-M", "HAN", {
            'hidden_channels': M, 'num_layers': 2, 'heads': 4,
            'dropout': 0.24, 'activation_type': 'prelu',
        }),
        ("HAN-L", "HAN", {
            'hidden_channels': L, 'num_layers': 2, 'heads': 8,
            'dropout': 0.24, 'activation_type': 'prelu',
        }),

        # --- SeHGNN S/M/L ---
        ("SeHGNN-S", "SeHGNN", {**SEHGNN_BASE, 'hidden_channels': S, 'heads': 2}),
        ("SeHGNN-M", "SeHGNN", {**SEHGNN_BASE, 'hidden_channels': M, 'heads': 4}),
        ("SeHGNN-L", "SeHGNN", {**SEHGNN_BASE, 'hidden_channels': L, 'heads': 8}),

        # ============================================================
        # PART 2: DEPTH COMPARISON (all at M=64 size)
        # ============================================================

        # --- GCN: 1 vs 2 layers ---
        ("GCN-1L", "GCN", {'hidden_channels': M, 'num_layers': 1}),

        # --- GraphSAGE: 1 vs 2 layers ---
        ("Sage-1L", "SageGNN", {'hidden_channels': M, 'num_layers': 1}),

        # --- HAN: 1 vs 2 layers ---
        ("HAN-1L", "HAN", {
            'hidden_channels': M, 'num_layers': 1, 'heads': 4,
            'dropout': 0.24, 'activation_type': 'prelu',
        }),

        # --- SeHGNN: multi-hop pre-aggregation ---
        ("SeHGNN-2hop", "SeHGNN", {**SEHGNN_BASE, 'hidden_channels': M, 'heads': 4, 'num_hops': 2}),
        ("SeHGNN-3hop", "SeHGNN", {**SEHGNN_BASE, 'hidden_channels': M, 'heads': 4, 'num_hops': 3}),

        # ============================================================
        # PART 3: METAPATH COUNT (SeHGNN-M, varying max_metapaths)
        # ============================================================
        ("SeHGNN-5mp", "SeHGNN", {**SEHGNN_BASE, 'hidden_channels': M, 'heads': 4},
            {'metapath_discovery.automatic.max_metapaths': 5}),

        ("SeHGNN-10mp", "SeHGNN", {**SEHGNN_BASE, 'hidden_channels': M, 'heads': 4},
            {'metapath_discovery.automatic.max_metapaths': 10}),

        ("SeHGNN-20mp", "SeHGNN", {**SEHGNN_BASE, 'hidden_channels': M, 'heads': 4},
            {'metapath_discovery.automatic.max_metapaths': 20}),

        ("SeHGNN-30mp", "SeHGNN", {**SEHGNN_BASE, 'hidden_channels': M, 'heads': 4},
            {'metapath_discovery.automatic.max_metapaths': 30}),
    ]

    results = []
    for entry in models:
        label, model_name, overrides = entry[0], entry[1], entry[2]
        cfg_overrides = entry[3] if len(entry) > 3 else None

        print(f"\n{'='*60}")
        print(f"Benchmarking: {label} ({model_name})")
        print(f"  Overrides: {overrides}")
        if cfg_overrides:
            print(f"  Config overrides: {cfg_overrides}")
        print(f"{'='*60}")

        try:
            res = benchmark_model(config, model_name, overrides, graph_data, device, epochs=EPOCHS, config_overrides=cfg_overrides)
            res['label'] = label
            res['overrides'] = overrides
            results.append(res)

            print(f"  Parameters: {format_params(res['trainable_params'])} trainable / {format_params(res['total_params'])} total")
            print(f"  Training: {res['train_time_per_epoch']:.2f}s/epoch ({res['train_time_total']:.1f}s total for {EPOCHS} epochs)")
            print(f"  Inference: {res['inference_mean']*1000:.1f} +/- {res['inference_std']*1000:.1f} ms")
            if res.get('peak_mem_train_gb'):
                print(f"  Peak GPU memory: {res['peak_mem_train_gb']:.2f} GB (train), {res['peak_mem_inf_gb']:.2f} GB (inference)")
            if res.get('num_metapaths'):
                print(f"  Metapath channels: {res['num_metapaths']}")
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()

        # Clear GPU memory between models
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # Print summary table — efficiency
    print(f"\n\n{'='*130}")
    print("BENCHMARK RESULTS — EFFICIENCY")
    print(f"{'='*130}")
    print(f"{'Model':<14} {'Hidden':>6} {'Heads':>5} {'MPs':>4} {'Params':>10} {'Train/epoch':>12} {'Inference':>18} {'GPU Train':>10} {'GPU Inf':>9}")
    print(f"{'-'*100}")
    for r in results:
        h = r['overrides'].get('hidden_channels', '?')
        heads = r['overrides'].get('heads', '-')
        mps = r.get('num_metapaths', '-')
        inf_str = f"{r['inference_mean']*1000:.1f} +/- {r['inference_std']*1000:.1f} ms"
        mem_train = f"{r['peak_mem_train_gb']:.1f} GB" if r.get('peak_mem_train_gb') else '-'
        mem_inf = f"{r['peak_mem_inf_gb']:.1f} GB" if r.get('peak_mem_inf_gb') else '-'
        print(f"{r['label']:<14} {h:>6} {str(heads):>5} {str(mps):>4} {format_params(r['trainable_params']):>10} {r['train_time_per_epoch']:>10.2f}s {inf_str:>18} {mem_train:>10} {mem_inf:>9}")

    # Print summary table — test quality metrics
    print(f"\n\n{'='*130}")
    print("BENCHMARK RESULTS — TEST METRICS")
    print(f"{'='*130}")
    print(f"{'Model':<14} {'AUC-ROC Mom':>12} {'AUC-PR Mom':>11} {'AUC-ROC Liq':>12} {'AUC-PR Liq':>11} {'F1 Mom':>7} {'F1 Liq':>7}")
    print(f"{'-'*80}")
    for r in results:
        tm = r.get('test_metrics', {})
        auc_roc_mom = tm.get('test_auc_roc_mom', tm.get('auc_roc_mom'))
        auc_pr_mom  = tm.get('test_auc_pr_mom',  tm.get('auc_pr_mom'))
        auc_roc_liq = tm.get('test_auc_roc_liq', tm.get('auc_roc_liq'))
        auc_pr_liq  = tm.get('test_auc_pr_liq',  tm.get('auc_pr_liq'))
        f1_mom      = tm.get('test_f1_mom',       tm.get('f1_mom'))
        f1_liq      = tm.get('test_f1_liq',       tm.get('f1_liq'))
        fmt = lambda v: f"{v:.4f}" if v is not None else '-'
        print(f"{r['label']:<14} {fmt(auc_roc_mom):>12} {fmt(auc_pr_mom):>11} {fmt(auc_roc_liq):>12} {fmt(auc_pr_liq):>11} {fmt(f1_mom):>7} {fmt(f1_liq):>7}")

    # Save results as JSON
    out_path = project_root / "outputs" / "benchmark_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
