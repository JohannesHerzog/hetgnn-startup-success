"""Entry point: preprocess startup_nodes.csv and train XGBoost models.

Usage:
    python src/main.py
    python src/main.py --config path/to/config.yaml

After training, run inference with:
    python scripts/predict.py
"""
import sys
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train XGBoost models on startup_nodes.csv")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    np.random.seed(config["seed"])

    from src.ml.preprocessing import preprocess
    from src.ml.train import train

    print("=" * 50)
    print("PREPROCESSING")
    print("=" * 50)
    data = preprocess(config)

    print("\n" + "=" * 50)
    print("TRAINING")
    print("=" * 50)
    train(data, config)

    print("\n" + "=" * 50)
    print("DONE")
    print("=" * 50)
    print(f"Models: {config['paths']['models_dir']}/")
    print("Predict: python scripts/predict.py")


if __name__ == "__main__":
    main()
