# Training Guide

## Quick Start

### 1. Set API Key

```bash
# Windows PowerShell
$env:GROQ_API_KEY='your_key_here'

# Linux/Mac
export GROQ_API_KEY='your_key_here'
```

### 2. Basic Training

```bash
python train.py --episodes 600 --device cuda
```

### 3. Custom Reward Weights

```bash
# Increase novelty reward to encourage more fact sharing
python train.py --episodes 600 --w-novelty 1.0 --w-engagement 1.0

# Reduce question spam
python train.py --episodes 600 --w-question-spam-penalty -0.3
```

## Command Line Arguments

### Core Training
- `--episodes`: Number of training episodes (default: 600)
- `--turns`: Max turns per episode (default: 40)
- `--lr`: Learning rate (default: 3e-4)
- `--gamma`: Discount factor (default: 0.99)
- `--device`: Device for training (`cpu` or `cuda`, default: `cpu`)
- `--name`: Experiment name (optional identifier)

### Reward Weights (per paper formalization)

The reward function is: **R_t = w_e·r^eng + w_n·r^nov + w_r·r_resp + w_c·r_conclude + penalties**

- `--w-engagement`: Weight for engagement reward (default: 1.0)
- `--w-novelty`: Weight for novelty reward (default: 0.5)
- `--w-responsiveness`: Weight for responsiveness reward (default: 0.25)
- `--w-conclude`: Weight for conclude bonus (default: 0.2)
- `--w-transition-penalty`: Weight for transition penalty (default: -0.2)
- `--w-spam-penalty`: Weight for spam penalty (default: -0.1)
- `--w-question-spam-penalty`: Weight for question spam penalty (default: -0.15)

### Novelty Reward Parameters
- `--novelty-per-fact`: Base reward per new fact (default: 0.15)
- `--novelty-explain-bonus`: Bonus for using ExplainNewFact (default: 0.1)

### Visualization & Evaluation
- `--map-interval`: Save map visualization every N episodes (default: 50, set to 0 to disable)
- `--save-map-frames`: Save map snapshots at EVERY turn (creates many files, use sparingly)
- `--live-map-display`: Show live map windows during training

### Debugging
- `--show-prompts`: Show LLM prompts during training
- `--force-option`: Force agent to always choose this option (testing only)
- `--force-subaction`: Force agent to always choose this subaction (testing only)
- `--enable-live-monitor`: Enable live training monitor
- `--verbose`: Verbose output mode

## Examples

### Test Run (1 minute)
```bash
python train.py --episodes 3 --turns 10 --name "test" --device cuda
```

### Full Training (5-6 hours)
```bash
python train.py --episodes 500 --turns 40 --name "thesis" --device cuda
```

### Tune for More Facts
```bash
python train.py --episodes 600 --w-novelty 1.0 --novelty-per-fact 0.2 --device cuda
```

### Save Maps Every Turn (All Episodes)
```bash
python train.py --episodes 600 --save-map-frames --device cuda
```
**Warning:** This creates many files (turn × episode count). Use sparingly.

## Output Structure

Results saved to: `training_logs/experiments/exp_XXX_<name>/`

- `models/` - Trained model
- `logs/` - All metrics (JSON)
- `maps/` - Map visualizations
- `detailed_logs/` - Full episode logs (prompts, dialogues, states)
- `parameterization_results/` - Reward weight analysis
- `evaluation/` - Comprehensive evaluation plots (always generated)

## Cost

- Test (3 episodes): ~$0.01
- Full (500 episodes): ~$1-2

## Troubleshooting

**Training stops?** → Check Groq spending limit at console.groq.com  
**GPU not used?** → Run: `python -c "import torch; print(torch.cuda.is_available())"`  
**Evaluation failed?** → Check that training completed successfully
