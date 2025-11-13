# Quick Start: Hypothesis Testing

## 1. Train All Experiments

Run all experiments in sequence:

```bash
python experiments/run_all_experiments.py --episodes 600 --device cuda
```

Or train individually:

```bash
# Baseline
python train.py --episodes 600 --name baseline --device cuda

# H5: State Ablation
python experiments/h5_state_ablation/train.py --episodes 600 --name h5 --device cuda

# H6: Transition Reward
python experiments/h6_transition_reward/train.py --episodes 600 --name h6 --device cuda
```

## 2. Evaluate Each Experiment

```bash
# Find experiment directories
BASELINE_DIR="training_logs/experiments/YYYYMMDD/exp_XXX_baseline_*"
H5_DIR="training_logs/experiments/YYYYMMDD/major_XXX_h5_*"
H6_DIR="training_logs/experiments/YYYYMMDD/major_XXX_h6_*"

# Evaluate
python experiments/h5_state_ablation/evaluate.py --experiment-dir $H5_DIR
python experiments/h6_transition_reward/evaluate.py --experiment-dir $H6_DIR
```

## 3. Generate Comparison Report

```python
from experiments.shared.comparison_tools import HypothesisComparator
from pathlib import Path

# Set results directory (where evaluation outputs are saved)
results_dir = Path('experiments/results')
results_dir.mkdir(exist_ok=True)

# Generate comparison
comparator = HypothesisComparator(results_dir)
report = comparator.generate_comparison_report(
    results_dir / 'comparison_report.json'
)

print("✅ Comparison report saved!")
```

## 4. View Results

- Individual metrics: `experiments/results/<variant>_metrics.json`
- Comparison report: `experiments/results/comparison_report.json`
- Text summary: `experiments/results/comparison_report.txt`

## Expected Output

Each experiment creates:
- Training logs in `training_logs/experiments/YYYYMMDD/`
- Evaluation metrics in experiment's `evaluation/` directory
- Comparison reports in `experiments/results/`

## Tips

- Use **same hyperparameters** (episodes, turns, lr, gamma) for fair comparison
- Run experiments on **same hardware** for consistent timing
- Save experiment directories for reproducibility
- Check `experiments/README.md` for detailed documentation

