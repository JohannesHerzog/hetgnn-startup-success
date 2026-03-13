"""Model explainability via Captum-based feature attribution on GNN predictions."""
import csv
import os
import random

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F
from torch_geometric.explain import Explainer
import wandb

from .local_captum.captum_explainer import CaptumExplainer

EXPLAIN_SEED = 42


def _set_deterministic_seed(seed=EXPLAIN_SEED):
    """Set all random seeds and enable deterministic CUDA ops for reproducible explanations."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def sample_nodes_to_explain(data, sample_size=25, config=None):
    """
    Samples nodes to explain according to config['explain']['sampling_method'].

    Supported methods:
      - 'random': uniform random sampling (default)
      - 'stratified': class-balanced sampling by true labels (if available)
    """
    # Determine sampling method
    sampling_method = "random"
    if config is not None:
        sampling_method = (
            config.get("explain", {}).get("sampling_method", "random").lower()
        )

    # Get test indices
    startup_test_indices = data["startup"].test_mask.nonzero(as_tuple=True)[0]
    n_available = len(startup_test_indices)

    if n_available == 0:
        print("No test nodes available. Skipping explanation.")
        return None

    # Clamp sample size if necessary
    if sample_size > n_available:
        print(
            f"Warning: sample_size ({sample_size}) > available test nodes ({n_available}). Clamping to max."
        )
        sample_size = n_available

    # --- STRATIFIED SAMPLING ---
    if sampling_method == "stratified":
        try:
            labels = data["startup"].y[startup_test_indices]
            unique_labels = labels.unique()
            n_classes = len(unique_labels)

            if n_classes == 0:
                raise ValueError("No unique labels found for stratified sampling.")

            # Compute roughly balanced per-class quota
            per_class = max(1, sample_size // n_classes)
            sampled_nodes = []

            for label in unique_labels:
                class_mask = labels == label
                class_indices = startup_test_indices[class_mask]
                k = min(per_class, len(class_indices))
                perm = torch.randperm(len(class_indices))
                sampled_nodes.append(class_indices[perm[:k]])

            sampled_nodes = torch.cat(sampled_nodes)

            # If we over-sampled slightly, trim
            if len(sampled_nodes) > sample_size:
                sampled_nodes = sampled_nodes[:sample_size]

            print(
                f"Stratified sampling: {len(sampled_nodes)} nodes from {n_classes} classes."
            )
            return sampled_nodes

        except Exception as e:
            print(f"Stratified sampling failed ({e}); falling back to random.")

    # --- RANDOM SAMPLING (default or fallback) ---
    perm = torch.randperm(n_available)
    sampled_nodes = startup_test_indices[perm[:sample_size]]
    print(f"Random sampling: selected {len(sampled_nodes)} nodes.")
    return sampled_nodes


def make_explainer(wrapped_model, method_params, model_config):
    return Explainer(
        model=wrapped_model,
        algorithm=CaptumExplainer(**method_params),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=model_config
    )


class WrappedModel(torch.nn.Module):
    def __init__(self, full_model, get_logits_fn):
        super().__init__()
        self.full_model = full_model
        self.get_logits_fn = get_logits_fn

    def forward(self, x_dict, edge_index_dict, **kwargs):
        logits = self.get_logits_fn(self.full_model, x_dict, edge_index_dict)
        return logits


def get_binary_logits(full_model, x_dict, edge_index_dict):
    output_dict = full_model(x_dict, edge_index_dict)
    
    # Handle SeHGNN Masked Multi Task format
    if "out_mom" in output_dict:
        logits = output_dict["out_mom"]
    # Handle Multi-Task format
    elif "binary_output" in output_dict:
        logits = output_dict["binary_output"]["startup"]
    # Handle Standard Hetero format
    elif "startup" in output_dict:
        logits = output_dict["startup"]["output"]
    else:
        raise KeyError(f"Could not find logits in model output. Keys: {output_dict.keys()}")

    if isinstance(logits, dict):
        logits = logits["output"]
        
    # Check shape to decide between sigmoid and softmax
    if logits.dim() == 1 or logits.size(-1) == 1:
        probs = torch.sigmoid(logits.view(-1))
    else:
        # Assuming index 1 is positive class
        probs = F.softmax(logits, dim=-1)[:, 1]
        
    return probs


def get_multi_logits(full_model, x_dict, edge_index_dict):
    output_dict = full_model(x_dict, edge_index_dict)["startup"]
    logits = output_dict["output"]
    probs = F.softmax(logits, dim=-1)
    return probs


def get_binary_part_logits(full_model, x_dict, edge_index_dict):
    logits = full_model(x_dict, edge_index_dict)["binary_output"]["startup"]
    probs = torch.sigmoid(logits.view(-1))
    return probs


def get_multi_part_logits(full_model, x_dict, edge_index_dict):
    logits = full_model(x_dict, edge_index_dict)["multi_class_output"]["startup"]
    probs = F.softmax(logits, dim=-1)
    return probs


def get_masked_multi_task_mom_logits(full_model, x_dict, edge_index_dict):
    out = full_model(x_dict, edge_index_dict)
    logits = out["out_mom"]
    probs = torch.sigmoid(logits.view(-1))
    return probs


def get_masked_multi_task_liq_logits(full_model, x_dict, edge_index_dict):
    out = full_model(x_dict, edge_index_dict)
    logits = out["out_liq"]
    probs = torch.sigmoid(logits.view(-1))
    return probs


def create_improved_feature_importance_plot(explanation, feat_labels, path, top_k=20, title=None):
    """
    Create an improved feature importance plot with better label handling.
    Shows features from all node types that are important for startup prediction.
    """
    all_features = []
    all_labels = []
    all_raw_features = []
    
    # Collect features from all node types
    for node_type, node_mask in explanation.node_mask_dict.items():
        if node_mask is None:
            continue

        # Calculate feature importance
        # Abs importance for ranking magnitude
        abs_importance = node_mask.abs().mean(dim=0).cpu().numpy()
        # Raw importance for direction (+/-)
        raw_importance = node_mask.mean(dim=0).cpu().numpy()
        
        # Get labels for this node type
        type_labels = feat_labels.get(node_type, [f'{node_type}_feature_{i}' for i in range(len(abs_importance))])
        
        # Add node type prefix to distinguish features from different types
        if node_type != 'startup':  # Don't prefix startup features to keep them clean
            prefixed_labels = [f"{node_type}:{label}" for label in type_labels]
        else:
            prefixed_labels = type_labels
        
        all_features.extend(abs_importance)
        all_raw_features.extend(raw_importance)
        all_labels.extend(prefixed_labels)
    
    if not all_features:
        print("Warning: No feature importance data found")
        return
    
    all_features = np.array(all_features)
    all_raw_features = np.array(all_raw_features)
    
    # Get top-k most important features across all node types
    top_indices = np.argsort(all_features)[-top_k:]
    top_importance = all_features[top_indices]
    top_raw = all_raw_features[top_indices]
    top_labels = [all_labels[i] for i in top_indices]
    
    short_labels = []
    for label, raw_val in zip(top_labels, top_raw):
        sign = "(+)" if raw_val >= 0 else "(-)"
        short_labels.append(f"{label} {sign}")
    
    # Thesis figure style (matches generate_thesis_figures.py)
    _THESIS_RCPARAMS = {
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    }

    # Node type colors (matching graph schema figure)
    node_type_colors = {
        'startup': '#4682B4',     # steelblue
        'investor': '#FF8C00',    # darkorange
        'founder': '#228B22',     # forestgreen
        'city': '#DC143C',        # crimson
        'university': '#800080',  # purple
        'sector': '#A52A2A',      # brown
    }

    # Color bars by node type
    colors = []
    for label in top_labels:
        if label.startswith('startup:') or ':' not in label:
            colors.append(node_type_colors['startup'])
        elif label.startswith('investor:'):
            colors.append(node_type_colors['investor'])
        elif label.startswith('founder:'):
            colors.append(node_type_colors['founder'])
        elif label.startswith('city:'):
            colors.append(node_type_colors['city'])
        elif label.startswith('university:'):
            colors.append(node_type_colors['university'])
        elif label.startswith('sector:'):
            colors.append(node_type_colors['sector'])
        else:
            colors.append('gray')

    # Normalize x-axis to avoid scientific notation
    max_val = top_importance.max()
    if max_val > 0:
        exponent = int(np.floor(np.log10(max_val)))
        scale = 10 ** exponent
        plot_importance = top_importance / scale
        x_label = f"Mean |Attribution| (×10$^{{{exponent}}}$)"
    else:
        plot_importance = top_importance
        x_label = "Mean |Attribution|"

    with plt.rc_context(_THESIS_RCPARAMS):
        fig, ax = plt.subplots(figsize=(7, max(5, top_k * 0.28)))

        y_pos = np.arange(len(top_labels))
        ax.barh(y_pos, plot_importance, alpha=0.85, color=colors, edgecolor='white', linewidth=0.3)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(short_labels)
        ax.set_xlabel(x_label)
        ax.grid(axis='x', alpha=0.2, linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Legend for present node types
        present_types = set()
        for label in top_labels:
            for node_type in node_type_colors.keys():
                if label.startswith(f'{node_type}:') or (node_type == 'startup' and ':' not in label):
                    present_types.add(node_type)
                    break

        legend_elements = [
            patches.Patch(color=node_type_colors[nt], label=nt.capitalize())
            for nt in ['startup', 'investor', 'founder', 'city', 'university', 'sector']
            if nt in present_types
        ]
        if legend_elements:
            ax.legend(handles=legend_elements, loc='lower right', framealpha=0.8)

        if title:
            ax.set_title(title, pad=10)

        fig.tight_layout()
        fig.savefig(path, facecolor='white', edgecolor='none')
        plt.close(fig)

    # Save attribution data as CSV for reproducible regeneration
    csv_path = str(path).rsplit('.', 1)[0] + '_data.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['feature', 'abs_importance', 'raw_importance'])
        for label, abs_val, raw_val in zip(top_labels, top_importance, top_raw):
            writer.writerow([label, abs_val, raw_val])

    print(f"Saved improved feature importance plot with {len(top_labels)} features from {len(present_types)} node types")
    print(f"Saved attribution data to {csv_path}")


def call_explanation(sampled_nodes, data, explainer, explain_path, mode, config=None):
    print(f"Explaining {len(sampled_nodes)} sampled nodes ...")
    hetero_explanation = explainer(
        x=data.x_dict,
        edge_index=data.edge_index_dict,
        index=sampled_nodes
    )

    os.makedirs(explain_path, exist_ok=True)

    # Generate both versions for comparison
    original_feature_importance_path = f"{explain_path}/startup_feature_importance_{mode}_original.pdf"
    improved_feature_importance_path = f"{explain_path}/startup_feature_importance_{mode}_improved.pdf"

    print(f"Saving original feature importance plot to {original_feature_importance_path}")
    # Original PyTorch Geometric visualization
    hetero_explanation.visualize_feature_importance(
        path=original_feature_importance_path, feat_labels=data.feat_labels, top_k=30
    )

    print(f"Saving improved feature importance plot to {improved_feature_importance_path}")
    # Improved custom visualization
    mode_titles = {
        "mom_task": "Feature Attribution — Next Funding Round (Prediction)",
        "liq_task": "Feature Attribution — Exit (Prediction)",
    }
    create_improved_feature_importance_plot(
        hetero_explanation,
        data.feat_labels,
        improved_feature_importance_path,
        top_k=30,
        title=mode_titles.get(mode),
    )

    use_wandb = config["wandb"]["enabled"] if config else False
    if use_wandb:
        wandb.log({
            f"explanation/startup_feature_importance_{mode}_original": wandb.Image(original_feature_importance_path),
            f"explanation/startup_feature_importance_{mode}_improved": wandb.Image(improved_feature_importance_path)
        })
    


def explain_model(model, data, explain_path, target_mode, sample_size, method, config=None):
    """
    Explains a model's predictions on a heterogeneous graph and calculates
    explanation quality metrics, including a characterization curve.
    """
    if "RandomBaseline" in str(model.__class__):
        print("Skipping explanation for RandomBaseline model (no gradients).")
        return

    _set_deterministic_seed()

    print("\n" + "=" * 50)
    print("EXPLANATION")
    print("=" * 50)

    # Swap features to test set features if available
    # This is CRITICAL because data['startup'].x contains training features (where test nodes are zeroed/imputed)
    # We need the actual test features (x_test_mask) for the explainer to work correctly.
    original_x = None
    if hasattr(data["startup"], "x_test_mask"):
        print("Swapping features to x_test_mask for explanation...")
        original_x = data["startup"].x
        data["startup"].x = data["startup"].x_test_mask

    sampled_nodes = sample_nodes_to_explain(data, sample_size)

    if target_mode == "binary_prediction":
        wrapped_model = WrappedModel(model, get_binary_logits)
        model_config = dict(
            mode="binary_classification",
            task_level="node",
            return_type="probs",
        )
        explainer = make_explainer(wrapped_model, method, model_config)
        call_explanation(
            sampled_nodes, data, explainer, explain_path, "binary_prediction", config=config
        )

    elif target_mode == "multi_prediction":
        wrapped_model = WrappedModel(model, get_multi_logits)
        model_config = dict(
            mode="multiclass_classification",
            task_level="node",
            return_type="probs",
        )
        explainer = make_explainer(wrapped_model, method, model_config)
        call_explanation(
            sampled_nodes, data, explainer, explain_path, "multi_prediction", config=config
        )

    elif target_mode == "multi_task":
        binary_model = WrappedModel(model, get_binary_part_logits)
        multi_model = WrappedModel(model, get_multi_part_logits)

        binary_config = dict(
            mode="binary_classification",
            task_level="node",
            return_type="probs",
        )
        multi_config = dict(
            mode="multiclass_classification",
            task_level="node",
            return_type="probs",
        )

        binary_explainer = make_explainer(binary_model, method, binary_config)
        multi_explainer = make_explainer(multi_model, method, multi_config)

        call_explanation(
            sampled_nodes, data, binary_explainer, explain_path, "binary_task", config=config
        )
        call_explanation(
            sampled_nodes, data, multi_explainer, explain_path, "multi_task", config=config
        )

    elif target_mode == "masked_multi_task":
        mom_model = WrappedModel(model, get_masked_multi_task_mom_logits)
        liq_model = WrappedModel(model, get_masked_multi_task_liq_logits)

        mom_config = dict(
            mode="binary_classification",
            task_level="node",
            return_type="probs",
        )
        liq_config = dict(
            mode="binary_classification",
            task_level="node",
            return_type="probs",
        )

        mom_explainer = make_explainer(mom_model, method, mom_config)
        liq_explainer = make_explainer(liq_model, method, liq_config)

        call_explanation(
            sampled_nodes, data, mom_explainer, explain_path, "mom_task", config=config
        )
        call_explanation(
            sampled_nodes, data, liq_explainer, explain_path, "liq_task", config=config
        )
    # Restore original features
    if original_x is not None:
        print("Restoring original features...")
        data["startup"].x = original_x


def explain_single_node(node_idx, data, explainer, explain_path, mode, model=None, title=None):
    """
    Explain a single node's prediction.
    """
    _set_deterministic_seed()
    print(f"Explaining single node {node_idx} ...")

    # Enable gradient calculation for explanation
    torch.set_grad_enabled(True)
    
    # Create single-node index tensor
    target_index = torch.tensor([node_idx], device=data["startup"].x.device, dtype=torch.long)
    
    # Run Explainer
    hetero_explanation = explainer(
        x=data.x_dict,
        edge_index=data.edge_index_dict,
        index=target_index
    )

    os.makedirs(explain_path, exist_ok=True)
    
    # Explain file names
    prefix = f"startup_{node_idx}"
    improved_path = f"{explain_path}/{prefix}_feature_importance_{mode}_improved.pdf"

    print(f"Saving single node feature importance plot to {improved_path}")
    
    create_improved_feature_importance_plot(
        hetero_explanation,
        data.feat_labels,
        improved_path,
        top_k=20,
        title=title,
    )
    
    return hetero_explanation


def calc_hetero_fidelity(explainer, explanation, data, target_indices):
    """
    Calculates Fidelity+/- for a heterogeneous graph explanation.
    Handles both binary and multi-class classification.
    """

    def process_output(pred_probs):
        # Binary: shape [N] or [N, 1]; Multi-class: shape [N, C]
        if pred_probs.dim() == 1 or pred_probs.size(1) == 1:
            return (pred_probs > 0.5).long()
        else:
            return pred_probs.argmax(dim=1)

    y_hat_probs = explainer.get_prediction(data.x_dict, data.edge_index_dict)[
        target_indices
    ]
    y_hat = process_output(y_hat_probs)

    explain_y_hat_probs = explainer.get_masked_prediction(
        data.x_dict,
        data.edge_index_dict,
        explanation.node_mask_dict,
        explanation.edge_mask_dict,
    )[target_indices]
    explain_y_hat = process_output(explain_y_hat_probs)

    complement_node_mask = {
        key: 1.0 - mask for key, mask in explanation.node_mask_dict.items()
    }
    complement_edge_mask = {
        key: 1.0 - mask for key, mask in explanation.edge_mask_dict.items()
    }

    complement_y_hat_probs = explainer.get_masked_prediction(
        data.x_dict, data.edge_index_dict, complement_node_mask, complement_edge_mask
    )[target_indices]
    complement_y_hat = process_output(complement_y_hat_probs)

    true_labels = data["startup"].y[target_indices]

    pos_fidelity = (
        (y_hat == true_labels).float() - (complement_y_hat == true_labels).float()
    ).mean()
    neg_fidelity = (
        ((y_hat == true_labels).float() - (explain_y_hat == true_labels).float())
        .abs()
        .mean()
    )

    return pos_fidelity.item(), neg_fidelity.item()


def calc_hetero_unfaithfulness(
    explainer, explanation, data, target_indices, top_k: int = None
) -> float:
    """
    Calculates the unfaithfulness score for a heterogeneous graph explanation,
    based on KL divergence between original and masked prediction distributions.

    Args:
        explainer: The Explainer instance.
        explanation: The Explanation object containing node/edge masks.
        data: The full heterogeneous graph data object.
        target_indices: Indices of the target node type (e.g., 'startup') to explain.
        top_k (int, optional): If set, will use only top-k important features for masking.

    Returns:
        float: Unfaithfulness score (lower is better).
    """

    # Extract relevant inputs
    x_dict = data.x_dict
    edge_index_dict = data.edge_index_dict
    node_mask_dict = explanation.node_mask_dict
    edge_mask_dict = explanation.edge_mask_dict

    # Get original prediction on full graph
    y = explainer.get_prediction(x_dict, edge_index_dict)  # shape: [num_nodes]
    y = y[target_indices]

    # Optionally apply top-k node feature selection per node type
    if top_k is not None:
        new_node_mask_dict = {}
        for node_type, node_mask in node_mask_dict.items():
            feat_importance = node_mask.sum(dim=0)  # [num_features]
            topk_index = feat_importance.topk(top_k).indices
            new_mask = torch.zeros_like(node_mask)
            new_mask[:, topk_index] = 1.0
            new_node_mask_dict[node_type] = new_mask
        node_mask_dict = new_node_mask_dict

    # Get masked prediction
    y_hat = explainer.get_masked_prediction(
        x_dict, edge_index_dict, node_mask_dict, edge_mask_dict
    )
    y_hat = y_hat[target_indices]
    p = torch.stack([1 - y, y], dim=-1)  # [P(class 0), P(class 1)]
    q = torch.stack([1 - y_hat, y_hat], dim=-1)
    kl_div = F.kl_div(p.log(), q, reduction="batchmean")  # or use log_target=False
    unfaithfulness_score = 1 - float(torch.exp(-kl_div))

    return unfaithfulness_score
