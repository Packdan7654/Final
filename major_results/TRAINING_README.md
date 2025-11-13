# Training Guide

This guide explains how to train models and where results are automatically saved.

## Quick Start

### Train Baseline Model

```bash
python train.py --episodes 500 --device cuda --experiment-type major
```

### Train Specific Hypothesis

```bash
# H1: Flat Actor-Critic
python experiments/h1_option_structure/train.py --episodes 500 --device cuda

# H3: Minimal Prompts
python experiments/h3_prompt_headers/train.py --episodes 500 --device cuda

# H5: State Ablation
python experiments/h5_state_ablation/train.py --episodes 500 --device cuda

# H6: No Transition Rewards
python experiments/h6_transition_reward/train.py --episodes 500 --device cuda

# H7: Hybrid BERT
python experiments/h7_hybrid_bert/train.py --episodes 500 --device cuda
```

### Train Multiple Variations

```bash
# Train baseline and H1
python train_all_variations.py --variations baseline h1 --episodes 500 --device cuda --no-confirm

# Train all variations
python train_all_variations.py --variations all --episodes 500 --device cuda --no-confirm

# Train specific variations
python train_all_variations.py --variations baseline h1 h3 --episodes 500 --device cuda --no-confirm
```

## Training Parameters

### Core Arguments

```bash
python train.py \
    --episodes 500 \              # Number of training episodes
    --turns 40 \                  # Max turns per episode
    --lr 0.0001 \                 # Learning rate
    --gamma 0.99 \                # Discount factor
    --device cuda \               # Device (cpu or cuda)
    --experiment-type major       # Save to major_results/
```

### Reward Configuration

```bash
python train.py \
    --w-engagement 1.0 \              # Engagement reward weight
    --novelty-per-fact 0.25 \         # Novelty reward per new fact
    --w-responsiveness 0.25 \         # Responsiveness reward weight
    --w-conclude 0.2 \                # Conclude bonus per exhibit
    --w-transition-insufficiency -0.20 # Transition penalty when < 2 facts
```

### Visualization Options

```bash
python train.py \
    --map-interval 10 \           # Generate map every N episodes (0 to disable)
    --save-map-frames            # Save map frames for every turn
```

### Verbosity

```bash
python train.py \
    --verbose                    # Show detailed per-turn output
    # (default: only episode progress)
```

## What Gets Saved Automatically

After training completes, the following are automatically saved:

### 1. Training Metrics (`metrics/`)
- `training_metrics.json` - Episode returns, lengths, coverage, facts, option usage
- `rl_metrics.json` - RL-specific metrics (gradient norms, TD errors, convergence data)
- `learning_curves.json` - Learning curve data (returns, losses, entropies)
- `convergence_report.json` - Convergence analysis (episode, samples, time)

### 2. Basic Visualizations (`visualizations/basic/`)
- `learning_curve.png` - Episode returns with moving average
- `convergence_analysis.png` - Convergence analysis
- `rl_metrics_summary.png` - 2x2 grid: value loss, policy loss, entropy, value estimates

### 3. Evaluation Metrics (`evaluation/`)
- `*_metrics.json` - Hypothesis-specific evaluation metrics (auto-generated)

### 4. Training Artifacts (`training/`)
- `logs/` - All training logs and metrics
- `models/` - Final trained models
- `checkpoints/` - Model checkpoints (saved every 50 episodes)
- `maps/` - Map visualizations (if `--map-interval` specified)
- `metadata.json` - Experiment metadata

### 5. Documentation
- `README.md` - Model description and configuration
- `validation_report.txt` - Metrics validation report

## Output Location

Results are automatically organized in:
```
major_results/YYYYMMDD_NNN_modelname/
```

Where:
- `YYYYMMDD` = Date (e.g., 20251112)
- `NNN` = Experiment number (001, 002, 003, ...)
- `modelname` = Model variation (baseline, h1_flat_policy, etc.)

## Example Training Session

```bash
# Train baseline with 500 episodes
python train.py --episodes 500 --device cuda --experiment-type major

# Output:
# - Training runs for 500 episodes
# - Metrics saved to training_logs/experiments/YYYYMMDD/major_001_baseline_.../
# - Results automatically organized to major_results/YYYYMMDD_001_baseline/
# - Basic plots generated in visualizations/basic/
# - Evaluation metrics saved to evaluation/
# - Validation report created
```

## Finding Your Results

```bash
# List all experiments
ls major_results/

# Find latest baseline
ls -t major_results/ | grep baseline | head -1

# View experiment README
cat major_results/20251112_001_baseline/README.md

# Check validation report
cat major_results/20251112_001_baseline/validation_report.txt
```

## Troubleshooting

### Training stops early
- Check `validation_report.txt` for missing files
- Review training logs in `training/logs/`
- Check for errors in console output

### Missing plots
- Basic plots are auto-generated during training
- Advanced plots require: `python tools/generate_major_results_visualizations.py --advanced`

### Missing evaluation metrics
- Evaluation runs automatically after training
- If missing, run manually: `python experiments/<hypothesis>/evaluate.py --experiment-dir <path>`

## Next Steps

After training:
1. Check `validation_report.txt` to verify all metrics are saved
2. Review plots in `visualizations/basic/`
3. Run advanced visualizations if needed (see [VISUALIZATION_README.md](VISUALIZATION_README.md))
4. Compare with other models (see [EVALUATION_README.md](EVALUATION_README.md))

