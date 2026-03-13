"""Entry point for GNN startup success prediction: config loading, CLI parsing, and training orchestration."""
import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import copy

import torch
import torch.backends.cudnn
from torch_geometric import seed_everything
import wandb
import yaml

from src.ml.preprocessing import perform_preprocessing
from src.ml.visualize import visualize_graph
from src.ml.feature_visualization import visualize_graph_features, plot_nan_distribution, visualize_edge_statistics
from src.ml.heterophily_metrics import calculate_edge_homophily, calculate_class_homophily
from src.ml.metrics_export import save_error_report
from src.ml.train import Trainer
from src.ml.utils import load_config, deep_merge_dict, parse_config_overrides


def _safe_run(config):
    """Wrapper around run() that captures errors to local JSON for debugging."""
    try:
        run(config)
    except Exception as e:
        print(f"\n{'='*50}")
        print(f"RUN FAILED: {type(e).__name__}: {e}")
        print(f"{'='*50}")
        try:
            output_dir = config.get("output_dir", "outputs")
            results_dir = os.path.join(output_dir, "results")
            save_error_report(error=e, config=config, output_base_dir=results_dir)
        except Exception as save_err:
            print(f"Warning: could not save error report: {save_err}")
        raise  # Re-raise so wandb/caller still sees the failure


def _apply_embeddings_to_graph(graph_data, embeddings):
    """Replace startup node features with GNN embeddings for downstream use (e.g. XGBoost)."""
    graph_data["startup"].x = embeddings
    for attr in ("x_val_mask", "x_test_mask", "x_test_mask_original"):
        if hasattr(graph_data["startup"], attr):
            setattr(graph_data["startup"], attr, embeddings.clone())
    print("Replaced startup features with GNN embeddings.")


def run(config):
    seed = config["seed"]
    seed_everything(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    graph_data, _ = perform_preprocessing(
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

    if config["visualize"].get("feature_visualization", False):
        plot_nan_distribution(graph_data, node_type="startup")

    if config["visualize"]["enabled"] and not config["wandb"].get("use_sweep", False):
        print("\n" + "=" * 50)
        print("VISUALIZATION")
        print("=" * 50)

        visualize_graph(
            graph_data=graph_data,
            output_file=config["visualize"]["output_file"],
            visible_node_types=set(config["visualize"]["visible_node_types"]),
            show_labels=config["visualize"]["show_labels"],
            enable_physics=config["visualize"]["enable_physics"],
            max_nodes=config["visualize"].get("max_nodes", 1000),
            sample_method=config["visualize"].get("sample_method", "degree_based"),
            show_features=config["visualize"].get("show_features", False),
            max_features=config["visualize"].get("max_features", 10),
            use_masks=config["visualize"].get("use_masks", False),
            included_masks=config["visualize"].get("included_masks", ["train", "val", "test"]),
        )

    if not config["wandb"].get("use_sweep", False):
        if config["visualize"].get("feature_visualization", False):
            visualize_graph_features(graph_data)

        if config["visualize"].get("edge_visualization", False):
            visualize_edge_statistics(graph_data)

    preprocess_only_path = config.get("_preprocess_only")
    if preprocess_only_path:
        os.makedirs(os.path.dirname(preprocess_only_path) or ".", exist_ok=True)
        print(f"Saving preprocessed graph data to {preprocess_only_path}...")
        torch.save(graph_data, preprocess_only_path)
        print(f"Done. Graph saved ({len(graph_data.node_types)} node types, {len(graph_data.edge_types)} edge types)")
        return

    output_dir = config.get("_output_dir") or "outputs/pipeline_state"
    os.makedirs(output_dir, exist_ok=True)
    config["output_dir"] = output_dir

    graph_path = os.path.join(output_dir, "graph_data.pt")
    print(f"Saving preprocessed graph data to {graph_path}...")
    torch.save(graph_data, graph_path)

    trainer = Trainer(graph_data, config)

    if config["train"].get("use_gnn", True):
        print("\n" + "=" * 50)
        print("TRAINING GNN")
        print("=" * 50)
        trainer.train()

        if config.get("xgboost", {}).get("enabled", False) and config.get("xgboost", {}).get("use_gnn_embeddings", False):
            embedding_path = config.get("xgboost", {}).get("embedding_path", "outputs/gnn_embeddings.pt")
            embeddings = trainer.get_all_embeddings()
            os.makedirs(os.path.dirname(embedding_path), exist_ok=True)
            torch.save(embeddings, embedding_path)
            print(f"Saved GNN embeddings to {embedding_path} (Shape: {embeddings.shape})")
            _apply_embeddings_to_graph(graph_data, embeddings)

    if config.get("xgboost", {}).get("enabled", False):
        print("\n" + "=" * 50)
        print("TRAINING XGBOOST")
        print("=" * 50)

        if not config["train"].get("use_gnn", True) and config.get("xgboost", {}).get("use_gnn_embeddings", False):
            embedding_path = config.get("xgboost", {}).get("embedding_path", "outputs/gnn_embeddings.pt")
            if os.path.exists(embedding_path):
                embeddings = torch.load(embedding_path, weights_only=False)
                print(f"Loaded GNN embeddings from {embedding_path} (Shape: {embeddings.shape})")
                _apply_embeddings_to_graph(graph_data, embeddings)
            else:
                print(f"Warning: Embedding file {embedding_path} not found. Using default features.")

        xgb_config = copy.deepcopy(config)
        xgb_config["train"]["model"] = "XGBoost"
        xgb_config.setdefault("models", {})
        xgb_params = config.get("xgboost", {}).get("params", {})
        xgb_config["models"]["XGBoost"] = xgb_params

        print(f"Initializing Trainer with model='XGBoost' and params: {xgb_params}")
        xgb_trainer = Trainer(graph_data, xgb_config)
        xgb_trainer.train()

    if config.get("analysis", {}).get("enable_homophily_analysis", False):
        print("\n" + "=" * 50)
        print("HOMOPHILY ANALYSIS")
        print("=" * 50)

        if 'startup' in graph_data.node_types and hasattr(graph_data['startup'], 'y'):
            y = graph_data['startup'].y

            targets_map = {}
            if y is not None:
                if y.ndim > 1 and y.shape[1] >= 2:
                    targets_map["Momentum"] = y[:, 0]
                    targets_map["Liquidity"] = y[:, 1]
                else:
                    targets_map["Target"] = y if y.ndim == 1 else y[:, 0]

            if targets_map:
                for edge_type in graph_data.edge_types:
                    src_type, rel, dst_type = edge_type
                    if src_type == 'startup' and dst_type == 'startup':
                        edge_index = graph_data[edge_type].edge_index
                        for target_name, target_y in targets_map.items():
                            mshr = calculate_edge_homophily(edge_index, target_y)
                            class_hom = calculate_class_homophily(edge_index, target_y)
                            print(f"{rel} ({target_name}): MSHR={mshr:.4f}, ClassHom={class_hom:.4f}")
            else:
                print("Skipping homophily: No valid targets found.")
        else:
            print("Skipping homophily analysis: Startup nodes or labels not found.")

    print("\n" + "=" * 50)
    print("TESTING")
    print("=" * 50)
    trainer.evaluate_test()

    if config["wandb"]["enabled"] and wandb.run is not None:
        run_id = wandb.run.id
        project = wandb.run.project or "unknown"
        persistent_dir = os.path.join("outputs", "checkpoints", project, run_id)
        os.makedirs(persistent_dir, exist_ok=True)
        persistent_path = os.path.join(persistent_dir, "best_model.pt")
        trainer.save_checkpoint(persistent_path)
        print(f"Saved persistent checkpoint: {persistent_path}")

    if config["wandb"]["enabled"]:
        wandb.finish()


def main():
    base_config = load_config()

    # Parse CLI overrides (any --dot.notation value args) and special flags
    config_updates, special = parse_config_overrides(sys.argv[1:], base_config)
    config = deep_merge_dict(base_config, config_updates)

    # Handle --preprocess-only: save graph and exit (no training, no wandb)
    if special["preprocess_only"]:
        config["_preprocess_only"] = special["preprocess_only"]
        config["wandb"]["enabled"] = False
        run(config)
        sys.exit(0)

    # Handle --output-dir
    if special["output_dir"]:
        config["_output_dir"] = special["output_dir"]

    experiment_id = special["experiment_id"]

    if not config.get("wandb", {}).get("enabled", False):
        _safe_run(config)
        return

    # WandB sweep mode: create sweep and run agent
    if config["wandb"].get("use_sweep", False):
        wandb.init(
            project=config["wandb"]["project"],
            name=config["wandb"].get("name"),
            tags=config["wandb"].get("tags"),
            notes=config["wandb"].get("notes"),
            config=config,
        )

        with open(config["wandb"]["sweep_config_path"], "r") as f:
            sweep_cfg = yaml.safe_load(f)

        sweep_id = wandb.sweep(sweep_cfg, project=config["wandb"]["project"])

        def sweep_run():
            base = load_config()
            full = deep_merge_dict(base, wandb.config.as_dict())
            _safe_run(full)

        wandb.agent(sweep_id, function=sweep_run, count=None)
        return

    # Standard wandb run
    tags = list(config["wandb"].get("tags") or [])
    if experiment_id:
        tags.append(f"exp:{experiment_id}")

    wandb.init(
        project=config["wandb"]["project"],
        name=config["wandb"].get("name"),
        tags=tags or None,
        notes=config["wandb"].get("notes"),
    )

    # Merge sweep parameters if this is a wandb agent run
    if wandb.run:
        sweep_params = dict(wandb.config)
        if sweep_params:
            config = deep_merge_dict(config, sweep_params)
        wandb.config.update(config, allow_val_change=True)
        config["wandb"]["enabled"] = True

    _safe_run(config)


if __name__ == "__main__":
    main()