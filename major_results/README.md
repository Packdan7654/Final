# Major Results Directory

This directory contains all organized results for thesis experiments. Each experiment run gets its own dated, numbered folder for clear organization and easy tracking.

## Directory Structure

Each experiment is stored in a folder named: **`YYYYMMDD_NNN_modelname`**

Where:
- **YYYYMMDD**: Date of the experiment (e.g., 20251112)
- **NNN**: Experiment number for that model type on that date (001, 002, 003, ...)
- **modelname**: Model variation name (baseline, h1_flat_policy, etc.)

### Example Structure

```
major_results/
├── 20251112_001_baseline/
│   ├── README.md                    # Model description and configuration
│   ├── validation_report.txt        # Metrics validation report
│   ├── training/
│   │   ├── logs/                    # All training logs and metrics
│   │   ├── checkpoints/             # Model checkpoints for recovery
│   │   ├── models/                  # Final trained models
│   │   ├── maps/                    # Map visualizations
│   │   └── metadata.json            # Experiment metadata
│   ├── evaluation/                  # Evaluation metrics (auto-generated)
│   │   └── *_metrics.json
│   ├── visualizations/
│   │   ├── basic/                   # Auto-generated during training
│   │   ├── advanced/                # Generated on-demand
│   │   └── evaluation/              # Evaluation-specific plots
│   └── metrics/                     # Consolidated metrics files
│       ├── training_metrics.json
│       ├── rl_metrics.json
│       ├── learning_curves.json
│       └── convergence_report.json
└── 20251112_001_h1_flat_policy/
    └── ...
```

## Model Variations

| Model | Description | Example Folder |
|-------|-------------|----------------|
| **baseline** | Hierarchical Option-Critic with Standard BERT | `20251112_001_baseline/` |
| **h1_flat_policy** | Flat Actor-Critic (no hierarchical structure) | `20251112_001_h1_flat_policy/` |
| **h3_minimal_prompts** | Minimal Prompts (no structured headers) | `20251112_001_h3_minimal_prompts/` |
| **h5_state_ablation** | State Ablation (dialogue-act-only state) | `20251112_001_h5_state_ablation/` |
| **h6_transition_reward** | No Transition Rewards | `20251112_001_h6_transition_reward/` |
| **h7_hybrid_bert** | Hybrid BERT (standard for intent, DialogueBERT for context) | `20251112_001_h7_hybrid_bert/` |

## Quick Start

### Training a Model

```bash
# Train baseline model
python train.py --episodes 500 --device cuda --experiment-type major

# Train specific hypothesis
python experiments/h1_option_structure/train.py --episodes 500 --device cuda

# Train multiple variations
python train_all_variations.py --variations baseline h1 --episodes 500 --device cuda --no-confirm
```

**After training completes:**
- ✅ Metrics automatically saved to `metrics/`
- ✅ Basic plots automatically generated in `visualizations/basic/`
- ✅ Evaluation automatically runs and saves to `evaluation/`
- ✅ Results organized in `major_results/YYYYMMDD_NNN_modelname/`

### Running Evaluation Manually

```bash
# Evaluate specific experiment
python experiments/h1_option_structure/evaluate.py --experiment-dir major_results/20251112_001_h1_flat_policy --output-dir major_results/20251112_001_h1_flat_policy/evaluation

# Evaluate all experiments
python experiments/run_all_evaluations.py
```

### Generating Visualizations

```bash
# Generate advanced plots for latest baseline experiment
python tools/generate_major_results_visualizations.py --model baseline --advanced

# Generate for specific experiment
python tools/generate_major_results_visualizations.py --experiment 20251112_001_baseline --advanced

# Generate for all experiments
python tools/generate_major_results_visualizations.py --all --advanced
```

## Finding Experiments

### Command Line

```bash
# List all baseline experiments
ls major_results/ | grep baseline

# Find latest baseline
ls -t major_results/ | grep baseline | head -1

# Count experiments for a model
ls major_results/ | grep "h1_flat_policy" | wc -l
```

### Python

```python
from src.utils.major_results_manager import MajorResultsManager

manager = MajorResultsManager()

# All experiments
all_exps = manager.list_all_experiments()

# By model
baseline_exps = manager.list_experiments_by_model('baseline')

# Latest
latest_baseline = manager.get_latest_experiment('baseline')
```

## Automatic Organization

When you train a model, results are automatically organized:

1. **Detects model name** from experiment directory or metadata
2. **Creates dated folder** with auto-incrementing number (YYYYMMDD_NNN_modelname)
3. **Copies training results** from `training_logs/experiments/`
4. **Consolidates metrics** into `metrics/` folder
5. **Generates basic visualizations** automatically
6. **Runs evaluation** and saves to `evaluation/`
7. **Validates metrics** and creates validation report
8. **Creates README** with model description

## What Gets Saved Automatically

### During Training
- ✅ All training metrics JSON files
- ✅ All basic visualizations (`learning_curve.png`, `convergence_analysis.png`, `rl_metrics_summary.png`)
- ✅ Training logs
- ✅ Final model (saved to `training/models/`)
- ✅ Map visualizations (if `--map-interval` specified)

### After Training
- ✅ Results copied to `major_results/YYYYMMDD_NNN_modelname/`
- ✅ Metrics consolidated to `metrics/` folder
- ✅ Evaluation metrics saved to `evaluation/`
- ✅ Validation report created

## Documentation

- **[TRAINING_README.md](TRAINING_README.md)** - How to train models
- **[EVALUATION_README.md](EVALUATION_README.md)** - How to run evaluations
- **[VISUALIZATION_README.md](VISUALIZATION_README.md)** - How to generate plots

## Notes

- **Training logs** (`training_logs/experiments/`) remain the source of truth for raw training data
- **Major results** (`major_results/`) is the organized, thesis-ready location
- **Each run is unique**: Multiple runs of the same model get different numbers (001, 002, 003, ...)
- **Chronological ordering**: Experiments are naturally sorted by date and number
- **Never overwrites**: Old experiments are preserved
