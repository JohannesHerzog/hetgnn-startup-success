"""Case study utilities for loading trained models and performing inference on individual startups."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F
from src.ml.train import Trainer
from src.ml.utils import load_config
from src.ml.preprocessing import perform_preprocessing
from src.ml.graph_assembler import create_graph
from src.ml.explain import explain_single_node, make_explainer, get_binary_logits, WrappedModel
from src.ml.visualize import create_pyvis_network, visualize_embedding_neighborhood
from scripts.competitor_retrieval import CompetitorRetriever
import os
import argparse
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

def load_graph_and_model(model_path=None, graph_path=None, state_dir=None, config_overrides=None):
    # 1. Load Data and Model
    config = load_config()
    # Force settings for inference
    config["train"]["epochs"] = 0
    config["explain"]["enabled"] = False # Disable auto-explain

    # Apply config overrides (e.g. max_metapaths to match checkpoint architecture)
    if config_overrides:
        for key_path, value in config_overrides.items():
            parts = key_path.split(".")
            d = config
            for p in parts[:-1]:
                d = d.setdefault(p, {})
            d[parts[-1]] = value

    # Load persistence paths
    if state_dir is None:
        state_dir = "outputs/pipeline_state"
    if graph_path is None:
        graph_path = os.path.join(state_dir, "graph_data.pt")
    if model_path is None:
        # Prefer last_model.pt (more stable, not overwritten by sweeps)
        last_path = os.path.join(state_dir, "models", "last_model.pt")
        best_path = os.path.join(state_dir, "models", "best_model.pt")
        model_path = last_path if os.path.exists(last_path) else best_path
    
    if not os.path.exists(graph_path):
        raise FileNotFoundError(f"Graph data not found at {graph_path}! Please run 'python src/main.py' first.")

    print(f"Loading graph data from {graph_path}...")
    graph_data = torch.load(graph_path, weights_only=False)
    
    print("Initializing model...")
    trainer = Trainer(graph_data=graph_data, config=config)
    
    # Move data to correct device (Model is on device, Data must be too for full-batch inference)
    print(f"   Moving data to {trainer.device}...")
    trainer.data = trainer.data.to(trainer.device)
    
    if os.path.exists(model_path):
        print(f"Found checkpoint at {model_path}, loading...")
        trainer.load_checkpoint(model_path)
    else:
        print("WARNING: No checkpoint found. Model is using random initialization!")
        
    return trainer

def collect_neighbors(data, center_idx, n_hops=1, max_per_type=50):
    """Collect n-hop neighbors of a startup node from the heterogeneous graph."""
    neighbors = {"startup": {center_idx}}
    frontier = {"startup": {center_idx}}  # nodes to expand from

    for hop in range(n_hops):
        next_frontier = {}
        for edge_type in data.edge_types:
            src, rel, dst = edge_type
            edge_index = data[edge_type].edge_index

            # Expand from frontier nodes of type src
            if src in frontier and frontier[src]:
                src_set = frontier[src]
                for node_idx in src_set:
                    mask = edge_index[0] == node_idx
                    connected = edge_index[1][mask].unique().tolist()
                    for c in connected:
                        if dst not in neighbors:
                            neighbors[dst] = set()
                        if c not in neighbors.get(dst, set()):
                            neighbors.setdefault(dst, set()).add(c)
                            next_frontier.setdefault(dst, set()).add(c)

            # Reverse direction
            if dst in frontier and frontier[dst]:
                dst_set = frontier[dst]
                for node_idx in dst_set:
                    mask = edge_index[1] == node_idx
                    connected = edge_index[0][mask].unique().tolist()
                    for c in connected:
                        if src not in neighbors:
                            neighbors[src] = set()
                        if c not in neighbors.get(src, set()):
                            neighbors.setdefault(src, set()).add(c)
                            next_frontier.setdefault(src, set()).add(c)

        frontier = next_frontier

    # Remove center from startup neighbors, cap per type
    neighbors["startup"].discard(center_idx)
    result = {}
    for ntype, idx_set in neighbors.items():
        idx_list = sorted(idx_set)
        if len(idx_list) > max_per_type:
            idx_list = idx_list[:max_per_type]
        if idx_list:
            result[ntype] = torch.tensor(idx_list, dtype=torch.long, device='cpu')

    return result


def _get_node_label(node_names_df, ntype, gidx, type_labels):
    """Look up a human-readable label for a node."""
    lbl = f"{type_labels.get(ntype, ntype)} {gidx}"
    if ntype in node_names_df:
        ndf = node_names_df[ntype]
        try:
            # Try exact "name" column first
            if "name" in ndf.columns:
                lbl = str(ndf.iloc[gidx]["name"])
            # Type-specific name columns (e.g. sector_name, city)
            else:
                name_cols = [c for c in ndf.columns if "name" in c.lower()]
                if name_cols:
                    lbl = str(ndf.iloc[gidx][name_cols[0]])
                elif "city" in ndf.columns:
                    lbl = str(ndf.iloc[gidx]["city"])
        except (IndexError, KeyError):
            pass
    return lbl


def plot_static_ego_graph(data, neighbors, node_scores_full, edge_scores_map,
                          best_node_idx, name, explain_path, metapath_names=None,
                          original_data=None, suffix=""):
    """Render a publication-quality static ego graph using matplotlib + networkx."""
    import matplotlib.colors as mcolors

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })

    # Color scheme
    colors = {
        "startup": "steelblue",
        "investor": "darkorange",
        "founder": "forestgreen",
        "city": "crimson",
        "university": "purple",
        "sector": "#95a5a6",
    }
    type_labels = {
        "startup": "Startup",
        "investor": "Investor",
        "founder": "Founder",
        "city": "City",
        "university": "University",
        "sector": "Sector",
    }

    # Build networkx graph
    G = nx.Graph()
    # Use original_data for node_names (clone() doesn't preserve custom attrs)
    names_source = original_data if original_data is not None else data
    node_names_df = getattr(names_source, 'node_names', {})

    # Center node
    center_id = f"startup_{best_node_idx}"
    G.add_node(center_id, ntype="startup", label=name, is_center=True)

    # Add neighbor nodes
    node_id_map = {}  # (ntype, global_idx) -> node_id
    node_id_map[("startup", best_node_idx)] = center_id

    for ntype, indices in neighbors.items():
        idx_list = indices.tolist() if torch.is_tensor(indices) else list(indices)
        for gidx in idx_list:
            if ntype == "startup" and gidx == best_node_idx:
                continue
            nid = f"{ntype}_{gidx}"
            lbl = _get_node_label(node_names_df, ntype, gidx, type_labels)
            G.add_node(nid, ntype=ntype, label=lbl, is_center=False)
            node_id_map[(ntype, gidx)] = nid

    # Add edges from the actual graph data
    for edge_type in data.edge_types:
        src_type, rel, dst_type = edge_type
        edge_index = data[edge_type].edge_index

        for s, d in zip(edge_index[0].tolist(), edge_index[1].tolist()):
            src_id = node_id_map.get((src_type, s))
            dst_id = node_id_map.get((dst_type, d))
            if src_id and dst_id and src_id != dst_id and G.has_node(src_id) and G.has_node(dst_id):
                score = edge_scores_map.get(edge_type, 0.5)
                G.add_edge(src_id, dst_id, rel=rel, score=score)

    # Remove isolated nodes (no edges to center's subgraph)
    isolates = list(nx.isolates(G))
    G.remove_nodes_from(isolates)

    print(f"   Graph has {len(G.nodes())} nodes, {len(G.edges())} edges")

    # Compute node importance scores
    max_score = 1e-10
    node_importance = {}
    for nid in G.nodes():
        ntype = G.nodes[nid]["ntype"]
        gidx = int(nid.split("_")[-1])
        if ntype in node_scores_full:
            scores = node_scores_full[ntype]
            if gidx < len(scores):
                imp = abs(scores[gidx].item())
            else:
                imp = 0.0
        else:
            imp = 0.0
        node_importance[nid] = imp
        if imp > max_score:
            max_score = imp

    # Layout: center node fixed at origin, tighter spacing
    n_nodes = len(G.nodes())
    k_spacing = 1.2 / (n_nodes ** 0.4 + 1) if n_nodes > 10 else 0.8
    pos = nx.spring_layout(G, k=k_spacing, iterations=200, seed=42,
                           fixed=[center_id], pos={center_id: (0, 0)},
                           scale=0.6)

    fig, ax = plt.subplots(figsize=(14, 12))

    # Draw edges
    edge_widths = []
    edge_colors_list = []
    max_edge_score = max(edge_scores_map.values()) if edge_scores_map else 1.0
    max_edge_score = max(max_edge_score, 1e-10)
    for u, v, edata in G.edges(data=True):
        s = edata.get("score", 0.5)
        edge_widths.append(2.0 + s / max_edge_score * 4.0)
        alpha = 0.25 + 0.35 * min(s / max_edge_score, 1.0)
        edge_colors_list.append((0.5, 0.5, 0.5, alpha))

    nx.draw_networkx_edges(G, pos, ax=ax, width=edge_widths, edge_color=edge_colors_list)

    # Draw nodes by type — bigger sizes
    for ntype in set(nx.get_node_attributes(G, "ntype").values()):
        nodes_of_type = [n for n in G.nodes() if G.nodes[n]["ntype"] == ntype]
        if not nodes_of_type:
            continue

        sizes = []
        node_colors_list = []
        for n in nodes_of_type:
            imp = node_importance[n] / max_score if max_score > 1e-10 else 0.0

            if G.nodes[n].get("is_center"):
                sizes.append(24000)
            else:
                # Base 6000, scale up to 18000 by importance
                sizes.append(6000 + imp ** 0.8 * 12000)

            node_colors_list.append(colors.get(ntype, "#95a5a6"))

        nx.draw_networkx_nodes(G, pos, nodelist=nodes_of_type, node_size=sizes,
                              node_color=node_colors_list, ax=ax,
                              edgecolors='white', linewidths=1.5, alpha=1.0)

    # Label all nodes (for small graphs) or top-N + center (for large)
    if n_nodes <= 15:
        label_nodes = set(G.nodes())
    else:
        top_n = 15
        sorted_nodes = sorted(node_importance.items(), key=lambda x: x[1], reverse=True)
        label_nodes = {center_id}
        for nid, _ in sorted_nodes[:top_n]:
            label_nodes.add(nid)

    # Draw name labels
    labels = {}
    for nid in label_nodes:
        lbl = G.nodes[nid].get("label", nid)
        if len(lbl) > 22:
            lbl = lbl[:20] + "..."
        labels[nid] = lbl

    nx.draw_networkx_labels(G, pos, labels, font_size=15, font_family="serif",
                           ax=ax, font_weight="bold")

    # Draw importance score annotations below each node
    for nid in label_nodes:
        imp = node_importance.get(nid, 0.0)
        x, y = pos[nid]
        # Format as plain decimal, not scientific notation
        if imp >= 0.01:
            score_str = f"{imp:.3f}"
        elif imp > 0:
            score_str = f"{imp:.4f}"
        else:
            score_str = "0"
        ax.annotate(score_str, (x, y), textcoords="offset points",
                   xytext=(0, -10), ha="center", va="top",
                   fontsize=16, fontstyle="italic", color="black",
                   fontfamily="serif")

    # Legend
    legend_handles = []
    present_types = set(nx.get_node_attributes(G, "ntype").values())
    for ntype in ["startup", "investor", "founder", "city", "university", "sector"]:
        if ntype in present_types:
            legend_handles.append(mpatches.Patch(
                color=colors.get(ntype, "#95a5a6"),
                label=type_labels.get(ntype, ntype)
            ))
    ax.legend(handles=legend_handles, loc="upper left", frameon=True, framealpha=0.9,
              edgecolor='#cccccc', fontsize=13)

    hop_label = suffix.replace("_", " ").strip() if suffix else ""
    title = f"Ego Graph: {name}" + (f" ({hop_label})" if hop_label else "")
    ax.set_title(title, fontsize=18, fontweight="bold", pad=20)
    ax.axis("off")

    # Expand axis limits so labels don't clip
    x_vals = [p[0] for p in pos.values()]
    y_vals = [p[1] for p in pos.values()]
    x_margin = (max(x_vals) - min(x_vals)) * 0.25
    y_margin = (max(y_vals) - min(y_vals)) * 0.15
    ax.set_xlim(min(x_vals) - x_margin, max(x_vals) + x_margin)
    ax.set_ylim(min(y_vals) - y_margin, max(y_vals) + y_margin)

    for spine in ax.spines.values():
        spine.set_visible(False)

    out_path = f"{explain_path}/startup_{best_node_idx}_ego_graph{suffix}.pdf"
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"   Saved static ego graph to {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Run Voize case study")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to model checkpoint")
    parser.add_argument("--graph-path", type=str, default=None,
                        help="Path to graph data .pt file")
    parser.add_argument("--max-metapaths", type=int, default=None,
                        help="Override max_metapaths to match checkpoint architecture")
    parser.add_argument("--num-hops", type=int, default=None,
                        help="Override SeHGNN num_hops to match checkpoint")
    args = parser.parse_args()

    print("Starting Single Startup Case Study...")

    config_overrides = {}
    if args.max_metapaths is not None:
        config_overrides["metapath_discovery.automatic.max_metapaths"] = args.max_metapaths
    if args.num_hops is not None:
        config_overrides["models.SeHGNN.num_hops"] = args.num_hops

    try:
        trainer = load_graph_and_model(
            model_path=args.model_path,
            graph_path=args.graph_path,
            config_overrides=config_overrides or None,
        )
    except Exception as e:
        print(e)
        return

    config = trainer.config

    # Run test evaluation to get predictions on test set
    print("Generating predictions...")
    trainer.model.eval()
    
    # We need predictions for ALL startups to find a candidate
    # Use the test set (masked)
    data = trainer.data
    
    with torch.no_grad():
        out = trainer.model(data.x_dict, data.edge_index_dict)
        
        # Determine output format
        if isinstance(out, dict) and "out_mom" in out:
             # SeHGNN Masked Multi Task format
             mom_out = out["out_mom"] # [N] (logits)
             liq_out = out["out_liq"] # [N] (logits)
             
             mom_probs = torch.sigmoid(mom_out)
             liq_probs = torch.sigmoid(liq_out)
             
             # Combined score for selection (e.g. geometric mean or simply sum)
             pred_probs = (mom_probs + liq_probs) / 2
        
        elif isinstance(out, dict) and "startup" in out:
             # Standard PyG Hetero format
             pred_probs = out["startup"]["output"]
             if pred_probs.size(1) > 1:
                 pred_probs = F.softmax(pred_probs, dim=1)[:, 1] # Probability of positive class
             else:
                 pred_probs = torch.sigmoid(pred_probs).squeeze()
             
             mom_probs = pred_probs
             liq_probs = pred_probs
        else:
             print("WARNING: Unknown output format, check case_study.py")
             return

    # Find interesting candidate
    # We want a startup that is PREDICTED to succeed (high prob) but maybe hasn't yet (mask=1? or check truth)
    # Let's just pick top prediction for now
    
    print("selecting candidate startup...")
    
    # Target UUID for Voize
    target_uuid = "5b8450df-dfb5-4d47-9168-4918c1aba3ac"
    
    # Try to find the target startup
    if hasattr(data, 'node_names') and 'startup' in data.node_names:
        startup_names = data.node_names['startup']
        # Check if we have UUIDs in the index or a column? 
        # Usually node_names has 'name' and uses uuid as index? 
        # Let's check if the index matches or if we need to search.
        # Assuming index is the UUID based on standard pipeline behavior
        
        # Try to find target UUID in index or column
        target_indices = []
        
        # Check explicit column first (safer)
        if 'startup_uuid' in startup_names.columns:
            matches = startup_names.index[startup_names['startup_uuid'] == target_uuid].tolist()
            target_indices.extend(matches)
        
        # Check index if no matches found yet
        if not target_indices and target_uuid in startup_names.index:
             loc = startup_names.index.get_loc(target_uuid)
             if isinstance(loc, int):
                 target_indices.append(loc)
             elif isinstance(loc, slice):
                 target_indices.extend(range(loc.start, loc.stop, loc.step or 1))
             else:
                 # Boolean array or similar
                 target_indices.extend(np.where(loc)[0].tolist())

        if target_indices:
             best_node_idx = target_indices[0] # Take first match
             best_prob = pred_probs[best_node_idx].item()
             name = startup_names.iloc[best_node_idx]['name'] if 'name' in startup_names.columns else "Unknown"
             print(f"Found target startup '{name}' (UUID: {target_uuid}) at index {best_node_idx}")
        else:
             print(f"WARNING: Target UUID {target_uuid} not found in dataset (Index or startup_uuid column). Falling back to top prediction.")
            # Fallback to top prediction
            # Get test mask if available, else usage all
             if "startup" in data and hasattr(data["startup"], "test_mask"):
                 mask = data["startup"].test_mask
             else:
                 mask = torch.ones_like(pred_probs, dtype=torch.bool)
                 
             # Filter by mask
             candidates = torch.where(mask)[0]
             candidate_probs = pred_probs[mask]
             
             # Sort
             sorted_indices = torch.argsort(candidate_probs, descending=True)
             
             # Pick top 1
             top_relative_idx = sorted_indices[0]
             best_node_idx = candidates[top_relative_idx].item()
             best_prob = candidate_probs[top_relative_idx].item()
             
             print(f"Top Candidate: Node {best_node_idx} with Prob: {best_prob:.4f}")

    else:
        # Fallback if no names available to lookup
        print("WARNING: No node names available for lookup. Falling back to top prediction.")
        # Get test mask if available, else usage all
        if "startup" in data and hasattr(data["startup"], "test_mask"):
             mask = data["startup"].test_mask
        else:
             mask = torch.ones_like(pred_probs, dtype=torch.bool)
             
        # Filter by mask
        candidates = torch.where(mask)[0]
        candidate_probs = pred_probs[mask]
        
        # Sort
        sorted_indices = torch.argsort(candidate_probs, descending=True)
        
        # Pick top 1
        top_relative_idx = sorted_indices[0]
        best_node_idx = candidates[top_relative_idx].item()
        best_prob = candidate_probs[top_relative_idx].item()

        print(f"Top Candidate: Node {best_node_idx} with Prob: {best_prob:.4f}")

    print(f"Selected candidate: Node {best_node_idx} with Prob: {best_prob:.4f}")
    
    node_idx = best_node_idx
    
    # Define explainer config
    # We need to know which task to explain. Let's explain Momentum for now.
    task_to_explain = "momentum" # or "liquidity"
    
    if hasattr(data, 'node_names') and 'startup' in data.node_names:
        name = data.node_names['startup'].iloc[best_node_idx]['name']
        print(f"   Name: {name}")

    # 3. Explain Feature Importance
    print("\nGenerating Feature Importance...")
    explain_path = "outputs/case_study"
    os.makedirs(explain_path, exist_ok=True)
    
    explain_config = config.get("explain", {})
    method = explain_config.get("method", "integrated_gradients")
    if isinstance(method, dict):
        # Fallback if config structure is unexpected
        print(f"WARNING: config['explain']['method'] is a dict: {method}. Using 'integrated_gradients'.")
        method = "integrated_gradients"
        
    method_params = explain_config.get(method, {})
    if "attribution_method" not in method_params and method == "integrated_gradients":
         method_params["attribution_method"] = "IntegratedGradients"
         
    model_config = dict(
        mode="binary_classification",
        task_level="node",
        return_type="probs",
    )
    
    # Wrap model for Captum
    wrapped_model = WrappedModel(trainer.model, get_binary_logits)
    explainer = make_explainer(wrapped_model, method_params, model_config)
    
    # Call our new function
    print("   Running Integrated Gradients...")
    explanation = explain_single_node(
        best_node_idx,
        data,
        explainer,
        explain_path,
        mode="case_study",
        model=trainer.model,
        title=f"Feature Attribution — {name}",
    )
    
    # 4. Extract Attention Weights (SeHGNN specific)
    print("\nExtracting Attention Weights...")
    # Re-run forward pass to get internal state
    trainer.model.eval()
    with torch.no_grad():
        out_dict = trainer.model(data.x_dict, data.edge_index_dict)
    
    if "attention_weights" in out_dict:
        attn_weights = out_dict["attention_weights"]
        metapath_names = out_dict.get("metapath_names", [])

        node_attn = attn_weights[best_node_idx]
        node_attn_avg = node_attn.mean(dim=0).cpu().numpy()
        
        plt.figure(figsize=(12, 10))
        # Use pcolormesh or imshow with proper extent to center pixels
        # imshow default extent is (-0.5, M-0.5, M-0.5, -0.5) usually
        im = plt.imshow(node_attn_avg, cmap='viridis', interpolation='nearest')
        plt.colorbar(im, label="Attention Weight")
        
        plt.title(f"Metapath Interaction Matrix for {name}\n(Row i attends to Column j)", fontsize=14)
        
        # Center ticks on pixels
        tick_locs = np.arange(len(metapath_names))
        plt.xticks(tick_locs, metapath_names, rotation=90, fontsize=10)
        plt.yticks(tick_locs, metapath_names, fontsize=10)
        
        plt.tight_layout()
        plt.savefig(f"{explain_path}/startup_{best_node_idx}_attention.pdf", dpi=300)
        print(f"   Saved attention plot to {explain_path}/startup_{best_node_idx}_attention.pdf")
        
        # Interpretation text
        print("\nAttention Interpretation:")
        print("   The matrix shows how different relational views (metapaths) interact.")
        print("   - High Diagonal: The metapath reinforces its own signal (strong independent feature).")
        print("   - High Off-Diagonal (Row i, Col j): Metapath i relies on context from Metapath j.")
        
        # Column sum = total attention received by each metapath (centrality proxy)
        agg_importance = node_attn_avg.sum(axis=0)
        sorted_imp_idx = np.argsort(agg_importance)[::-1]
        
        print("\nTop Metapaths by Received Attention (Centrality):")
        for i in range(min(5, len(metapath_names))):
            idx = sorted_imp_idx[i]
            print(f"   {i+1}. {metapath_names[idx]} (Score: {agg_importance[idx]:.4f})")
            
        # Plot Bar Chart of Importance
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(metapath_names)), agg_importance, color='teal')
        plt.title(f"Metapath Centrality (Received Attention) for {name}")
        plt.xticks(range(len(metapath_names)), metapath_names, rotation=90)
        plt.ylabel("Total Attention Received")
        plt.tight_layout()
        plt.savefig(f"{explain_path}/startup_{best_node_idx}_metapath_importance.pdf")
        print(f"   Saved Metapath Importance Bar Chart.")
        
    else:
        print("   (Model does not return attention weights, skipping)")
    
    # 5. Calculate Node Importance from HeteroExplanation
    # Sum feature importance for each node to get scalar node importance
    print("\nCalculating Node Importance Scores...")
    node_scores_full = {}
    
    if hasattr(explanation, 'node_mask_dict'):
        masks = explanation.node_mask_dict
        for ntype, mask_tensor in masks.items():
            # Sum across feature dimension (dim=1)
            # mask_tensor: [Num_Nodes, Num_Features]
            scores = mask_tensor.sum(dim=1).detach().cpu()
            
            # Normalize? Optional. Let's keep raw sums relative to each other.
            node_scores_full[ntype] = scores
            
            # Debug: Top 3 important nodes globally
            if ntype == 'investor':
                top_k = torch.topk(scores, min(3, len(scores)))
                print(f"   Top investors globally: indices={top_k.indices.tolist()}, values={top_k.values.tolist()}")

    # 5b. Prepare Edge Scores from Attention
    # We map "Metapath Importance" to "Edge Type Importance"
    # Metapaths are tuples: (src, rel, dst)
    # Edge types in PyG are also tuples: (src, rel, dst)
    # This mapping is direct for 1-hop metapaths.
    print("\nPreparing Edge Attention Scores...")
    edge_scores_map = {}
    if "attention_weights" in out_dict:
        # agg_importance has shape [Num_Metapaths], aligned with metapath_names
        for mp_idx, mp_name in enumerate(metapath_names):
            # mp_name is ('startup', 'early_stage_funded_by', 'investor') etc.
            # or a string if it was converted. Let's assume tuple or check.
            # In models.py we utilize tuples. 
            
            # The score for this metapath
            score = agg_importance[mp_idx]
            
            # If mp_name is a tuple, we can use it directly as key for edge_scores
            # (assuming PyG edge_types match exactly). 
            # Note: "self" metapath is not an edge.
            if mp_name == "self": continue
            
            edge_scores_map[mp_name] = score.item() if hasattr(score, 'item') else score
            
            # Special handling: "Symmetric" metapaths might appear differently?
            # Usually they are standard edge types.
    
    # 6. Ego Graph Visualization
    print("\nGenerating Ego Graph...")

    # Move full data to CPU for visualization
    print("   Moving data to CPU for visualization...")
    viz_data = data.clone().to('cpu')

    # 6a. 1-hop ego graph (PDF)
    print("\nGenerating 1-hop ego graph (PDF)...")
    neighbors_1hop = collect_neighbors(viz_data, best_node_idx, n_hops=1)
    plot_static_ego_graph(
        data=viz_data,
        neighbors=neighbors_1hop,
        node_scores_full=node_scores_full,
        edge_scores_map=edge_scores_map,
        best_node_idx=best_node_idx,
        name=name,
        explain_path=explain_path,
        original_data=data,
        suffix="_1hop",
    )

    # 6b. 2-hop ego graph (PDF)
    print("\nGenerating 2-hop ego graph (PDF)...")
    neighbors_2hop = collect_neighbors(viz_data, best_node_idx, n_hops=2, max_per_type=30)
    plot_static_ego_graph(
        data=viz_data,
        neighbors=neighbors_2hop,
        node_scores_full=node_scores_full,
        edge_scores_map=edge_scores_map,
        best_node_idx=best_node_idx,
        name=name,
        explain_path=explain_path,
        original_data=data,
        suffix="_2hop",
    )

    # 7. Embedding Neighborhood Visualization
    print("\nGenerating Embedding Neighborhood Visualization...")
    
    # Initialize Retriever
    retriever = CompetitorRetriever(trainer)
    
    # Get Top Neighbors (Competitors)
    # top_k=200 for good density
    print("   Retrieving neighbors...")
    neighbor_indices, neighbor_scores = retriever.retrieve(best_node_idx, method='gnn', top_k=200)
    
    print("\n   Neighbor Analysis:")
    print(f"   Top 5 Neighbor Scores: {neighbor_scores[:5]}")
    print(f"   Bottom 5 Neighbor Scores: {neighbor_scores[-5:]}")
    
    # Check for identical embeddings
    if retriever.gnn_embeddings is None:
        retriever._extract_gnn_embeddings()
        
    sub_embs = retriever.gnn_embeddings[neighbor_indices]
    center_emb = retriever.gnn_embeddings[best_node_idx]
    
    # Calculate distance from center manually to verify
    from sklearn.metrics.pairwise import cosine_similarity
    sims = cosine_similarity(center_emb.reshape(1, -1), sub_embs)[0]
    print(f"   Re-calculated Top 5 Similarities: {sims[:5]}")
    
    # Variance check
    variance = np.var(sub_embs, axis=0).mean()
    print(f"   Neighbor Embedding Variance: {variance:.6f} (Low variance = Collapse)")
    
    # Add some random context nodes for contrast (100 random)
    all_indices = list(range(len(retriever.df)))
    random_indices = np.random.choice(all_indices, 100, replace=False)
    
    # Combine: Center + Neighbors + Random
    viz_indices = [best_node_idx] + list(neighbor_indices) + list(random_indices)
    viz_indices = list(set(viz_indices)) # Unique
    
    # Extract Embeddings for these nodes
    if retriever.gnn_embeddings is None:
        retriever._extract_gnn_embeddings()
        
    subset_embeddings = retriever.gnn_embeddings[viz_indices]
    
    # Build Metadata
    viz_metadata = {}
    for i, global_idx in enumerate(viz_indices):
        # We need to look up info in retriever.df or raw_df
        # Use retriever helper logic if possible, or manual
        row = retriever.df.iloc[global_idx]
        curr_name = row.get('name', str(global_idx))
        
        # Try raw df for better metadata if available
        meta_sector = "Unknown"
        meta_status = "Unknown"
        
        if retriever.raw_df is not None:
             uid = row.get('startup_uuid')
             if not uid: uid = row.get('items_id')
             if uid and uid in retriever.raw_df.index:
                 raw_row = retriever.raw_df.loc[uid]
                 meta_sector = str(raw_row.get('industry_groups', 'Unknown'))
                 if len(meta_sector) > 20: meta_sector = meta_sector[:17] + "..."
                 meta_status = str(raw_row.get('status', 'Unknown'))
        
        # Local index i maps to metadata
        viz_metadata[i] = {
            'name': curr_name,
            'sector': meta_sector,
            'status': meta_status
        }

    # Find local index of center node
    center_local_idx = viz_indices.index(best_node_idx)
    
    visualize_embedding_neighborhood(
        embeddings=subset_embeddings,
        node_indices=viz_indices,
        center_node_idx=center_local_idx,
        metadata=viz_metadata,
        output_path=f"{explain_path}/startup_{best_node_idx}_embedding_tsne.pdf",
        title=f"Embedding Landscape for {name}"
    )

    print("\nCase Study Complete!")

if __name__ == "__main__":
    main()
