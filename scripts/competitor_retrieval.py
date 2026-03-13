"""Competitor retrieval using GNN embeddings, text similarity, and hybrid fusion strategies."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F
from src.ml.train import Trainer
from src.ml.utils import load_config
import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import argparse
from scipy.stats import rankdata

class CompetitorRetriever:
    def __init__(self, trainer):
        self.trainer = trainer
        self.data = trainer.data
        self.config = trainer.config
        
        # Ensure data corresponds to startup dataframe
        if not hasattr(self.data['startup'], 'df'):
            print("WARNING: Startup DataFrame not attached to graph data.")
            self.df = None
        else:
            self.df = self.data['startup'].df
            
        # Load raw nodes for lookup (descriptions etc)
        self.raw_df = None
        # Try different paths
        raw_paths = ["data/graph/startup_nodes.csv", "../data/graph/startup_nodes.csv"]
        for p in raw_paths:
            if os.path.exists(p):
                print(f"Loading raw startup nodes from {p} for descriptions...")
                try:
                    self.raw_df = pd.read_csv(p)
                    # Index by UUID for fast lookup
                    uuid_col = 'startup_uuid'
                    if uuid_col in self.raw_df.columns:
                        self.raw_df.set_index(uuid_col, inplace=True)
                    else:
                        print(f"WARNING: {uuid_col} not found in raw CSV.")
                except Exception as e:
                    print(f"WARNING: Failed to load raw CSV: {e}")
                break
        
        # Cache for embeddings
        self.gnn_embeddings = None
        self.text_embeddings = None
        
    def find_startup(self, query_name):
        """
        Fuzzy search for startup by name.
        Returns list of (index, name, uuid, match_score)
        """
        if self.df is None:
            raise ValueError("Startup DataFrame missing.")
            
        # exact match
        query_lower = query_name.lower()
        
        matches = []
        for idx, row in self.df.iterrows():
            name = str(row.get('name', '')).lower()
            if query_lower in name:
                # Try to find UUID
                uid = row.get('startup_uuid')
                if not uid: uid = row.get('items_id', 'Unknown')
                
                matches.append((idx, row.get('name', 'Unknown'), uid)) 
                
        # Sort by length difference (heuristic for 'best match')
        matches.sort(key=lambda x: len(x[1]))
        return matches[:5]
        
    def _extract_text_embeddings(self):
        if self.text_embeddings is not None:
             return self.text_embeddings
            
        print("   Extracting Text Embeddings from DataFrame...")
        if self.df is None:
             raise ValueError("Startup DataFrame missing.")
             
        # Identify embedding columns
        emb_cols = [c for c in self.df.columns if c.startswith('desc_emb_')]
        if not emb_cols:
            print("No 'desc_emb_*' columns found in DataFrame. Baseline method unavailable.")
            return None
            
        # Sort to ensure indices 0, 1, 2...
        emb_cols.sort(key=lambda x: int(x.split('_')[-1]))
        
        # Extract as numpy matrix
        # Note: Index of df should align with node index if preprocessing did its job and we haven't shuffled weirdly.
        # graph_assembler.py:383 -> df.reset_index(drop=True), then df["startup_id"] = range...
        # So df index is the node index.
        
        self.text_embeddings = self.df[emb_cols].values # [N, D]
        
        # Handle NaNs (for rows without description)
        self.text_embeddings = np.nan_to_num(self.text_embeddings, nan=0.0)
        
        # Normalize for cosine similarity
        from sklearn.preprocessing import normalize
        self.text_embeddings = normalize(self.text_embeddings, axis=1)
        
        return self.text_embeddings
        
    def _extract_gnn_embeddings(self):
        if self.gnn_embeddings is not None:
            return self.gnn_embeddings
            
        print("   Running Inference to get GNN Embeddings...")
        self.trainer.model.eval()
        with torch.no_grad():
            out = self.trainer.model(self.data.x_dict, self.data.edge_index_dict)
            
            # Prefer retrieval_embedding if available (SimCLR/CLIP pattern)
            if "retrieval_embedding" in out and "startup" in out["retrieval_embedding"]:
                emb = out["retrieval_embedding"]["startup"]
                print("   Using retrieval_embedding (separate head)")
            elif "embedding" in out and "startup" in out["embedding"]:
                emb = out["embedding"]["startup"]
                print("   WARNING: Using task embedding (retrieval head disabled)")
            else:
                raise ValueError(f"Model output missing embeddings. Keys: {out.keys() if isinstance(out, dict) else type(out)}")
                 
            self.gnn_embeddings = emb.cpu().numpy()
            
            # Normalize
            from sklearn.preprocessing import normalize
            self.gnn_embeddings = normalize(self.gnn_embeddings, axis=1)
            
        return self.gnn_embeddings
        
    def _compute_rrf(self, map_of_scores, weights=None, k=60):
        """
        Compute Reciprocal Rank Fusion.
        map_of_scores: Dict of {method_name: scores_array}
        weights: Dict of {method_name: float_weight}
        """
        # Initialize with zeros (using size of first array)
        if not map_of_scores: return None
        first_scores = next(iter(map_of_scores.values()))
        final_score = np.zeros_like(first_scores)
        
        for name, scores in map_of_scores.items():
            # Rank descending (Higher score = Rank 1)
            ranks = rankdata(-scores, method='min')
            
            # Apply Weight
            w = weights.get(name, 1.0) if weights else 1.0
            final_score += w / (k + ranks)
            
        return final_score

    def _get_scores(self, query_node_idx, method):
        """
        Get raw similarity scores for a single method.
        """
        if method == 'text':
            embs = self._extract_text_embeddings()
        elif method == 'gnn':
            embs = self._extract_gnn_embeddings()
        else:
            raise ValueError(f"Unknown single method {method}")
            
        if embs is None: return None
        
        query_vec = embs[query_node_idx].reshape(1, -1)
        scores = cosine_similarity(query_vec, embs)[0]
        return scores
        
    def retrieve(self, query_node_idx, method='gnn', top_k=10, weights=None, fusion_strategy='rrf'):
        """
        Retrieve similar startups for a given node index.
        weights: dict, e.g. {'text': 4.0, 'gnn': 1.0}
        fusion_strategy: 'rrf' or 'avg' (weighted average)
        """
        # Load necessary embeddings lazily
        if method == 'text' or method == 'hybrid':
             _ = self._extract_text_embeddings()
        if method == 'gnn' or method == 'hybrid':
             _ = self._extract_gnn_embeddings()
             
        # Calculate Similarity
        if method == 'hybrid':
            score_map = {}
            
            # Get Component Scores
            s_text = self._get_scores(query_node_idx, 'text')
            if s_text is not None: score_map['text'] = s_text
            
            s_gnn = self._get_scores(query_node_idx, 'gnn')
            if s_gnn is not None: score_map['gnn'] = s_gnn
            
            if not score_map: return [], []
            
            # Fusion Strategy
            if fusion_strategy == 'avg':
                # Weighted Average
                first_scores = next(iter(score_map.values()))
                final_score = np.zeros_like(first_scores)
                
                for name, s in score_map.items():
                    w = weights.get(name, 1.0) if weights else 1.0
                    final_score += s * w
                
                # Normalize by sum of weights? Not strictly needed for ranking but cleaner
                total_w = sum([weights.get(n, 1.0) for n in score_map.keys()]) if weights else len(score_map)
                scores = final_score / total_w
                
            else:
                # RRF (Default)
                scores = self._compute_rrf(score_map, weights=weights)
            
        else:
            scores = self._get_scores(query_node_idx, method)
            if scores is None: return [], []
            
        # Self-retrieval removal
        scores[query_node_idx] = -1.0
        
        # Get Top K
        # argsort gives ascending, so we take last k and reverse
        top_indices = np.argsort(scores)[-top_k:][::-1]
        top_scores = scores[top_indices]
        
        return top_indices, top_scores

def print_results(retriever, query_idx, indices, scores, method_name):
    print(f"\nExample Retrieval Results ({method_name}):")
    print(f"{'Rank':<4} | {'Score':<6} | {'Name':<30} | {'Status':<15} | {'Sector':<20} | {'Description (Truncated)'}")
    print("-" * 130)
    
    for rank, (idx, score) in enumerate(zip(indices, scores)):
        row = retriever.df.iloc[idx]
        name = row.get('name', 'Unknown')
        
        # Metadata defaults
        status = row.get('status', 'Unknown')
        if 'future_status' in row: status = str(row['future_status'])
        
        sector = "Unknown"
        desc = str(row.get('description', ''))
        
        # Fallback to raw lookup for Metadata
        if retriever.raw_df is not None:
             uid = row.get('startup_uuid')
             if not uid: uid = row.get('items_id')
             
             if uid and uid in retriever.raw_df.index:
                 raw_row = retriever.raw_df.loc[uid]
                 
                 # Status Selection
                 # Try future_status -> dc_status -> operating_status
                 if pd.notnull(raw_row.get('future_status')): status = str(raw_row['future_status'])
                 elif pd.notnull(raw_row.get('dc_status')): status = str(raw_row['dc_status'])
                 
                 # Sector Selection
                 # Try industry_groups -> industries
                 if pd.notnull(raw_row.get('industry_groups')): 
                     sector = str(raw_row['industry_groups'])
                 elif pd.notnull(raw_row.get('industries')): 
                     sector = str(raw_row['industries'])
                     
                 # Clean up sector string (remove brackets/quotes if list)
                 sector = sector.replace("['", "").replace("']", "").replace("', '", ", ")
                 if len(sector) > 20: sector = sector[:17] + "..."
                     
                 # Desc
                 raw_desc = raw_row.get('description')
                 if (not desc or desc == 'nan' or desc == '') and pd.notnull(raw_desc):
                     desc = str(raw_desc)
        
        if not desc or desc == 'nan': desc = "[N/A]"
             
        desc_trunc = (desc[:75] + '...') if len(desc) > 75 else desc
        
        print(f"{rank+1:<4} | {score:.4f} | {name:<30} | {status:<15} | {sector:<20} | {desc_trunc}")


def main():
    parser = argparse.ArgumentParser(description="Startup Competitor Retrieval")
    parser.add_argument("--query", type=str, help="Name of startup to search for", default=None)
    parser.add_argument("--method", type=str, choices=['text', 'gnn', 'hybrid', 'all'], default='all', help="Retrieval method")
    parser.add_argument("--top_k", type=int, default=10, help="Number of results")
    parser.add_argument("--compare_strategies", action="store_true", help="Run comparison of Hybrid strategies")
    parser.add_argument("--weights", type=str, default=None, help="Weights in format 'text:1.0,gnn:0.5'")
    args = parser.parse_args()

    # Parse Weights
    weights = {'text': 1.0, 'gnn': 1.0}
    if args.weights:
        try:
            parts = args.weights.split(',')
            for p in parts:
                k, v = p.split(':')
                weights[k.strip()] = float(v)
            print(f" Using Custom Weights: {weights}")
        except Exception as e:
            print(f"WARNING: Error parsing weights: {e}")
            return

    print("Initializing Competitor Retrieval...")
    
    # 1. Load State
    config = load_config()
    state_dir = "outputs/pipeline_state"
    graph_path = os.path.join(state_dir, "graph_data.pt")
    model_path = os.path.join(state_dir, "models", "best_model.pt")
    
    if not os.path.exists(graph_path):
        print(f"Graph data not found at {graph_path}")
        return

    print("Loading graph data...")
    graph_data = torch.load(graph_path, weights_only=False)
    
    print("Loading model...")
    trainer = Trainer(graph_data=graph_data, config=config)
    trainer.data = trainer.data.to(trainer.device)
    
    if os.path.exists(model_path):
        trainer.load_checkpoint(model_path)
    else:
        print("WARNING: Model checkpoint not found, using random weights (Results will be garbage for GNN)")
        
    retriever = CompetitorRetriever(trainer)
    
    # 2. Select Query Node
    query_idx = None
    query_name = "Unknown"
    
    if args.query:
        matches = retriever.find_startup(args.query)
        if not matches:
            print(f"No startup found matching '{args.query}'")
            return
            
        print(f"Found {len(matches)} matches:")
        for i, (idx, name, uid) in enumerate(matches):
            print(f"   [{i}] {name} (ID: {uid})")
            
        # Default to first match
        query_idx = matches[0][0]
        query_name = matches[0][1]
        print(f"Selected: {query_name} (Node {query_idx})")
        
    else:
        # Check for case_study_uuid in config
        case_study_uuid = config.get('eval', {}).get('case_study_uuid')
        found_case_study = False
        
        if case_study_uuid:
            print(f"Found case_study_uuid in config: {case_study_uuid}")
            # Try to find in df
            # Check column 'startup_uuid'
            if 'startup_uuid' in retriever.df.columns:
                 matches = retriever.df.index[retriever.df['startup_uuid'] == case_study_uuid].tolist()
                 if matches:
                     query_idx = matches[0]
                     query_name = retriever.df.loc[query_idx].get('name', 'Unknown')
                     print(f"Selected Case Study Startup: {query_name} (Node {query_idx})")
                     found_case_study = True
            
            # Check index if not found
            if not found_case_study and case_study_uuid in retriever.df.index:
                 # Check if integer index or label
                 if isinstance(retriever.df.index, pd.RangeIndex) or isinstance(retriever.df.index, pd.Int64Index):
                     # If index is integer, we can't search by string UUID in it directly unless it's mixed
                     pass 
                 else:
                     # Assume index is UUIDs?
                     # pipeline usually resets index to int.
                     pass

        if not found_case_study:
            # Pick a random startup with a description embedding (fair to text method)
            print("No query provided and case study startup not found. Picking a random startup with description embedding...")
            
            # Check if embeddings exist
            emb_cols = [c for c in retriever.df.columns if c.startswith('desc_emb_')]
            if not emb_cols:
                 print("WARNING: No description embeddings found. Picking completely random node.")
                 valid_indices = retriever.df.index.tolist()
            else:
                 # Check if first embedding column is not NaN
                 has_desc = retriever.df[emb_cols[0]].notna()
                 valid_indices = np.where(has_desc)[0]
            
            import random
            if len(valid_indices) > 0:
                query_idx = random.choice(valid_indices)
            else:
                query_idx = random.randint(0, len(retriever.df)-1)    
                
            query_name = retriever.df.iloc[query_idx].get('name', 'Unknown')
            print(f"Selected Random: {query_name} (Node {query_idx})")
        
    # Print Query Info
    print(f"\nTarget Startup: {query_name}")
    
    # Get Description
    desc = "N/A"
    # Try graph df first
    if 'description' in retriever.df.columns:
        desc = retriever.df.iloc[query_idx].get('description', '')
    
    # Try lookup
    if (not desc or desc == "N/A" or pd.isna(desc)) and retriever.raw_df is not None:
        row = retriever.df.iloc[query_idx]
        uid = row.get('startup_uuid')
        if not uid: uid = row.get('items_id')
        
        if uid and uid in retriever.raw_df.index:
            desc = retriever.raw_df.loc[uid, 'description']

    desc_str = str(desc) if pd.notnull(desc) else "N/A"
    print(f"Description: {desc_str[:200]}...")
    
    # 3. Run Retrieval
    
    if args.compare_strategies:
        print("\nRunning Strategy Comparison...")
        experiments = [
            ('gnn', None, 'rrf'),
            ('text', None, 'rrf'),
            ('hybrid', {'text': 1.0, 'gnn': 1.0}, 'avg'),  # Old "Score Average"
            ('hybrid', {'text': 1.0, 'gnn': 1.0}, 'rrf'),  # RRF Baseline (Stage-First)
            ('hybrid', {'text': 4.0, 'gnn': 1.0}, 'rrf'),  # RRF Topic-First
        ]
        
        for m, w, strat in experiments:
            label = f"{m.upper()}"
            if m == 'hybrid':
                label += f" ({strat.upper()}"
                if w: label += f" | T:{w.get('text',1)} G:{w.get('gnn',1)}"
                label += ")"
                
            try:
                indices, scores = retriever.retrieve(query_idx, method=m, top_k=args.top_k, weights=w, fusion_strategy=strat)
                print_results(retriever, query_idx, indices, scores, method_name=label)
            except Exception as e:
                print(f"{label} failed: {e}")
                
    else:
        # Standard Run
        methods = ['text', 'gnn', 'hybrid'] if args.method == 'all' else [args.method]
        
        for m in methods:
            try:
                # Default strategy is RRF for hybrid unless args say otherwise (not exposed yet)
                # Using parsed weights
                indices, scores = retriever.retrieve(query_idx, method=m, top_k=args.top_k, weights=weights, fusion_strategy='rrf')
                print_results(retriever, query_idx, indices, scores, method_name=m.upper())
            except Exception as e:
                print(f"\nMethod {m.upper()} failed: {e}")

if __name__ == "__main__":
    # If running interactively without args, simulate args
    if len(sys.argv) == 1:
        # sys.argv.append("--help") # Un-comment to see help
        pass
        
    main()
