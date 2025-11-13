"""
Run matched training runs for the hierarchical (HRL) and flat RL variants and
produce directly comparable summaries.
"""

import argparse
import json
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from src.flat_rl.training_loop import FlatTrainingLoop
from src.training.training_loop import HRLTrainingLoop


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train HRL and Flat RL agents with identical settings and generate comparison summaries."
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=50,
        help="Number of training episodes for each agent.",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=20,
        help="Maximum turns per episode.",
    )
    parser.add_argument(
        "--knowledge-graph",
        type=str,
        default="museum_knowledge_graph.json",
        help="Path to the museum knowledge graph JSON.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for training ('cpu' or 'cuda').",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to store comparison outputs. Defaults to training_logs/comparisons/<timestamp>/",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for both agents.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose episode prints.",
    )
    return parser.parse_args()


def build_output_dir(path: str | None) -> Path:
    if path:
        base = Path(path)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = Path("training_logs") / "comparisons" / f"flat_vs_hrl_{timestamp}"
    base.mkdir(parents=True, exist_ok=True)
    return base


def run_training(loop_cls, label: str, base_kwargs: dict, output_root: Path):
    exp_dir = output_root / label
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Ensure experiment artefacts are scoped
    os.environ["EXPERIMENT_DIR"] = str(exp_dir)
    try:
        loop = loop_cls(**base_kwargs)
        loop.run_training()
        summary = loop.metrics_tracker.get_summary_statistics()
    finally:
        os.environ.pop("EXPERIMENT_DIR", None)

    # Persist summary snapshot
    summary_path = exp_dir / "metrics_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    return summary


def main():
    args = parse_args()

    output_dir = build_output_dir(args.output_dir)
    print(f"[+] Comparison outputs will be stored in: {output_dir}")

    base_kwargs = {
        "max_episodes": args.episodes,
        "max_turns_per_episode": args.max_turns,
        "knowledge_graph_path": args.knowledge_graph,
        "device": args.device,
        "enable_live_monitor": False,
        "save_metrics": True,
        "enable_map_viz": False,
        "save_map_frames": False,
        "live_map_display": False,
        "verbose": args.verbose,
    }

    compare_results = {}

    for label, loop_cls in [("hrl", HRLTrainingLoop), ("flat", FlatTrainingLoop)]:
        print(f"\n=== Running {label.upper()} training ===")
        set_global_seed(args.seed)
        summary = run_training(loop_cls, label, base_kwargs, output_dir)
        compare_results[label] = summary

    summary_path = output_dir / "comparison_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(compare_results, f, indent=2, sort_keys=True)

    print("\n=== Comparison Summary ===")
    metrics_to_show = [
        "total_episodes",
        "mean_return",
        "recent_mean_return",
        "mean_length",
        "mean_coverage",
        "mean_dwell",
    ]

    for metric in metrics_to_show:
        hrl_value = compare_results.get("hrl", {}).get(metric, "n/a")
        flat_value = compare_results.get("flat", {}).get(metric, "n/a")
        print(f"{metric:>20}: HRL={hrl_value} | Flat={flat_value}")

    print(f"\n[+] Detailed summaries saved to {summary_path}")


if __name__ == "__main__":
    main()


