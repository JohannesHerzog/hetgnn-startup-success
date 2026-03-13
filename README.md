# HetGNN Startup Success

Source code for the master's thesis:

> **Path-Aware Heterogeneous Graph Neural Networks for Explainable Startup Success Prediction and Competitor Retrieval**
> Cedric Lorenz, Hasso Plattner Institute, University of Potsdam, 2026

## Overview

We model the startup ecosystem as a heterogeneous graph with six node types (**Startup**, **Investor**, **Founder**, **City**, **University**, **Sector**) connected by edges encoding funding relationships (across 4 stages), team membership, industry classification, geographic location, founder education, and founder–investor professional ties. Startups are additionally linked by text-similarity edges derived from S-BERT embeddings. The full graph comprises ~1.1M nodes and ~2.2M edges.

We predict two complementary tasks:

- **Next Funding Round** — whether a startup will secure additional funding
- **Exit** — whether a startup will achieve an IPO or acquisition

Our best GNN, **SeHGNN** (Simple and Efficient Heterogeneous GNN), uses transformer-based metapath fusion to aggregate information across curated metapaths encoding investment co-participation, alumni networks, VC employment links, and board governance interlocks. Both tasks share a single heterogeneous graph encoder via masked multi-task learning, with task-specific prediction heads.

### Models

| Model | Type | Description |
|-------|------|-------------|
| **SeHGNN** | Heterogeneous GNN | Transformer-based metapath fusion (best performer) |
| **HAN** | Heterogeneous GNN | Hierarchical attention over metapaths |
| **GraphSAGE** | Homogeneous GNN | Inductive representation learning via sampling and aggregation (`SageGNN` in CLI) |
| **GCN** | Homogeneous GNN | Graph Convolutional Network (homogeneous baseline) |
| **MLP** | Neural Network | Feature-only baseline (no graph structure) |
| **XGBoost** | Gradient Boosting | Classical ML baseline |
| **LLM** | Language Model | Zero-shot LLM baseline (Llama 3 8B) |
| **Random** | Baseline | Random predictions, lower bound (`RandomBaseline` in CLI) |
| **Degree** | Baseline | Degree centrality as prediction score (`DegreeCentrality` in CLI) |

## Setup

### Prerequisites

- Python 3.11+
- CUDA-compatible GPU (tested with NVIDIA A40)
- ~16 GB GPU memory for SeHGNN training

### Installation

```bash
# Clone the repository
git clone https://github.com/cedric-lorenz/hetgnn-startup-success.git
cd hetgnn-startup-success

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

> **Note:** This project depends on [PyTorch Geometric](https://pyg.org/), which requires
> matching PyTorch and CUDA versions. If `pip install` fails for PyG packages, follow the
> [official installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

### Data

This project uses Crunchbase data, which is proprietary and cannot be redistributed.
To reproduce results, you need access to the [Crunchbase Data](https://data.crunchbase.com/)
and should place the CSV exports in `data/crunchbase/`.

The data pipeline in `src/data_engineering/` processes raw Crunchbase CSVs into the feature
matrices and graph structure used for training.

#### Synthetic Data (for testing without Crunchbase access)

To run the full pipeline without proprietary Crunchbase data, generate a synthetic dataset:

```bash
# Default: 10,000 startups
python scripts/generate_synthetic_data.py
```

This generates all node and edge types with realistic feature distributions matching the real Crunchbase data schema into `data/graph/`. After generation, train normally with `python src/main.py`.

**Note on synthetic data**: The synthetic dataset is intended for testing the pipeline end-to-end, not for meaningful evaluation. Some models may not learn from synthetic data due to the lack of realistic feature–label correlations. All models work correctly on real Crunchbase data.

## Usage

### Training

All configuration is in `config.yaml`. Parameters can be overridden via CLI using dot notation:

```bash
# Train SeHGNN (default) — full pipeline: data loading, graph construction,
# metapath materialization, training, evaluation, and metrics export
python src/main.py

# Override hyperparameters
python src/main.py --train.model SeHGNN --train.lr 0.01 --train.epochs 100

# Train other models
python src/main.py --train.model HAN      # Hierarchical Attention Network
python src/main.py --train.model SageGNN  # GraphSAGE
python src/main.py --train.model GCN      # Graph Convolutional Network
python src/main.py --train.model MLP      # Feature-only baseline
python src/main.py --train.model XGBoost  # Gradient boosting baseline

# Baselines
python src/main.py --train.model RandomBaseline       # Random predictions
python src/main.py --train.model DegreeCentrality     # Degree centrality baseline

# Zero-shot LLM baseline (Llama 3 8B)
python src/main.py --train.model LLM --models.LLM.prompt_features full

# Explainability via Integrated Gradients
python src/main.py --train.model SeHGNN --explain.enabled true --explain.sample_size 100
```

Each training run produces:
- Evaluation metrics (AUC-ROC, AUC-PR, F1, P@k) printed to stdout
- JSON metrics export in `outputs/results/<model>/<target_mode>/`
- Model checkpoint in `outputs/pipeline_state/models/`

> **Tip:** To run without a Weights & Biases account, set `WANDB_MODE=disabled` or add `--wandb.enabled false`.

### Analysis Scripts

After training, several scripts in `scripts/` can be used for post-hoc analysis:

```bash
# Compute graph statistics (node/edge counts, degree distributions, homophily)
python scripts/compute_graph_statistics.py

# Benchmark model architectures (parameter count, training/inference time)
python scripts/benchmark_models.py

# Single-startup case study (attention weights, ego graph, feature attribution)
python scripts/case_study.py

# Competitor retrieval via GNN embeddings
python scripts/competitor_retrieval.py

# Compare GNN vs text embeddings (clustering, similarity analysis)
python scripts/compare_embeddings.py

# Analyze GNN embedding space (pairwise similarity, UMAP visualization)
python scripts/analyze_embeddings.py

# Feature-level ablation analysis
python scripts/feature_level_analysis.py

# Generate publication-quality thesis figures
python scripts/generate_thesis_figures.py
```

> **Note:** Most scripts require a trained model checkpoint in `outputs/pipeline_state/`. Some additionally require Crunchbase CSVs in `data/crunchbase/` for metadata (organization names, sectors, geography).

### Testing

```bash
python -m pytest tests/ -v
```

## Project Structure

```
.
├── src/
│   ├── __init__.py
│   ├── main.py                    # Entry point: config loading, CLI, training orchestration
│   ├── ml/
│   │   ├── __init__.py
│   │   ├── models.py              # All model architectures (SeHGNN, HAN, GraphSAGE, GCN, MLP, XGBoost, baselines)
│   │   ├── train.py               # Training loop with early stopping, LR scheduling
│   │   ├── eval.py                # Evaluation: AUC-ROC/PR, F1, P@k, metrics export
│   │   ├── preprocessing.py       # Data loading, scaling, feature engineering
│   │   ├── graph_assembler.py     # Heterogeneous graph construction, metapath materialization
│   │   ├── explain.py             # Integrated Gradients explainability
│   │   ├── calibration.py         # Platt scaling, isotonic regression, ECE
│   │   ├── llm_predictor.py       # Zero-shot LLM baseline
│   │   ├── metapath_discovery.py  # Manual/automatic/hybrid metapath discovery
│   │   ├── metrics_export.py      # JSON export of evaluation metrics
│   │   ├── visualize.py           # Training curves, graph visualization
│   │   ├── imputation.py          # Split-specific missing value imputation
│   │   ├── heterophily_metrics.py # Node-level and edge-level homophily analysis
│   │   ├── downstream_analysis.py # VC portfolio simulation and ROI analysis
│   │   ├── feature_visualization.py # Feature distribution plots
│   │   ├── utils.py               # Config loading utilities
│   │   └── local_captum/          # Modified Captum explainer for heterogeneous graphs
│   └── data_engineering/          # Crunchbase data pipeline
│       ├── __init__.py
│       ├── data_pipeline.py       # Main ETL pipeline
│       ├── org_features.py        # Startup feature engineering
│       ├── person_features.py     # Founder feature engineering
│       ├── finance_features.py    # Funding round features
│       ├── city_features.py       # City-level features
│       ├── filtering.py           # Data quality filters
│       ├── targets.py             # Target variable definitions
│       └── aux_pipeline.py        # Auxiliary data processing
├── scripts/                       # Analysis, figure generation, and thesis utility scripts
├── tests/                         # Unit tests
├── config.yaml                    # Main configuration file
└── requirements.txt               # Python dependencies
```

## Attribution

The SeHGNN implementation adapts the architecture from:
> Yang, Z., et al. "Simple and Efficient Heterogeneous Graph Neural Network." *Proceedings of the AAAI Conference on Artificial Intelligence*, 2023.

The explainability module (`src/ml/local_captum/`) contains modified source code from [Captum](https://github.com/pytorch/captum) (BSD-3-Clause), extending `IntegratedGradients` and `CaptumExplainer` to support heterogeneous graph explanations.

## Citation

```bibtex
@mastersthesis{lorenz2026hetgnn,
  title   = {Path-Aware Heterogeneous Graph Neural Networks for Explainable
             Startup Success Prediction and Competitor Retrieval},
  author  = {Lorenz, Cedric},
  school  = {Hasso Plattner Institute, University of Potsdam},
  year    = {2026},
  type    = {Master's Thesis}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
