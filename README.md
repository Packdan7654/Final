# HRL Museum Dialogue Agent

Hierarchical and flat reinforcement learning for adaptive museum dialogue.

## Quick Start

### 1. Set API Key
```bash
$env:GROQ_API_KEY='your_key_here'  # Windows PowerShell
export GROQ_API_KEY='your_key_here'  # Linux/Mac
```

### 2. Train

**Single Model:**
```bash
# Baseline (default)
python train.py --episodes 500 --device cuda

# Specific variant
python train.py --variant h1 --episodes 500 --device cuda
python train.py --variant h3 --episodes 500 --device cuda
```

**All Variations (with auto-evaluation):**
```bash
python train_all_variations.py --episodes 500 --device cuda --no-confirm
```

### 3. Results

- **`train.py`**: Saves to `training_logs/experiments/YYYYMMDD/`
- **`train_all_variations.py`**: Saves to `major_results/YYYYMMDD_NNN_modelname/` (with auto-evaluation & plots)

---

## Command Reference

### train.py

**Basic:**
```bash
python train.py --episodes 500 --device cuda
```

**Variants:**
```bash
python train.py --variant baseline --episodes 500
python train.py --variant h1 --episodes 500
python train.py --variant h3 --episodes 500
python train.py --variant h5 --episodes 500
python train.py --variant h6 --episodes 500
python train.py --variant h7 --episodes 500
```

**Key Arguments:**
- `--episodes N` - Number of episodes (default: 500)
- `--device cuda|cpu` - Device (default: cpu)
- `--variant baseline|h1|h3|h5|h6|h7` - Model variant
- `--experiment-type major|minor` - Experiment type (both save to training_logs/)
- `--verbose` - Show detailed output

**Full reference:** See [TRAIN_COMMAND_REFERENCE.md](TRAIN_COMMAND_REFERENCE.md)

---

### train_all_variations.py

**Train all variations:**
```bash
python train_all_variations.py --episodes 500 --no-confirm
```

**Train specific variations:**
```bash
python train_all_variations.py --variations baseline h1 h3 --episodes 500
```

**Key Arguments:**
- `--episodes N` - Episodes per variation (default: 500)
- `--variations baseline h1 h3 ...` - Specific variations (default: all)
- `--device cuda|cpu` - Device (default: cuda)
- `--no-confirm` - Skip confirmation prompt

**What it does:**
- Trains all specified variations sequentially
- Saves to `major_results/` (not `training_logs/`)
- Automatically runs evaluation after each training
- Automatically generates all plots
- Organizes everything with validation

---

## Directory Structure

### train.py Output
```
training_logs/experiments/YYYYMMDD/
├── major_XXX_modelname_YYYYMMDD_HHMMSS/
│   ├── logs/
│   ├── maps/
│   ├── models/
│   └── metadata.json
└── exp_XXX_YYYYMMDD_HHMMSS/  # --experiment-type minor
```

### train_all_variations.py Output
```
major_results/
├── baseline/YYYYMMDD_NNN_baseline/
│   ├── metrics/          # Training metrics
│   ├── evaluation/      # Hypothesis metrics
│   ├── visualizations/  # All plots (auto-generated)
│   └── README.md
├── h1_flat_policy/...
└── ...
```

**Key Difference:**
- `train.py` → `training_logs/` (simple, no auto-evaluation)
- `train_all_variations.py` → `major_results/` (organized, auto-evaluation, auto-plots)

---

## Project Structure

```
Thesis_HRL/
├── train.py                    # Single model training
├── train_all_variations.py     # Batch training (all variants)
├── experiments/                # Hypothesis-specific code
│   ├── h1_option_structure/
│   ├── h3_prompt_headers/
│   ├── h5_state_ablation/
│   ├── h6_transition_reward/
│   └── h7_hybrid_bert/
├── src/                        # Core code
│   ├── agent/                  # RL agents
│   ├── environment/            # Environment
│   ├── training/               # Training loops
│   └── utils/                  # Utilities
├── training_logs/              # Individual training outputs
└── major_results/              # Batch training outputs (organized)
```

---

## Requirements

```bash
pip install -r requirements.txt
```

Key dependencies:
- PyTorch
- Transformers (for BERT)
- NumPy, Matplotlib, Seaborn
- Groq API (for LLM)

---

## Documentation

- **[TRAIN_COMMAND_REFERENCE.md](TRAIN_COMMAND_REFERENCE.md)** - Complete command reference
- **[major_results/README.md](major_results/README.md)** - Results directory guide

---

## Notes

- All `train.py` scripts save to `training_logs/` (both major and minor experiment types)
- Only `train_all_variations.py` saves to `major_results/` with automatic evaluation
- Evaluation and plots are automatically generated when using `train_all_variations.py`
- Individual training scripts do NOT automatically organize to `major_results/`
