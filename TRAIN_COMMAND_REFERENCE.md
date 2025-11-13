# Training Command Reference

Complete command reference for all training scripts.

---

## train.py

**Unified training script for all model variations.**

### Quick Examples

```bash
# Baseline (default)
python train.py --episodes 500 --device cuda

# Specific variant
python train.py --variant h1 --episodes 500 --device cuda
python train.py --variant h3 --episodes 500 --device cuda
```

### Arguments

| Argument | Options | Default | Description |
|----------|---------|---------|-------------|
| **Model Selection** |
| `--variant` | `baseline`, `h1`, `h3`, `h5`, `h6`, `h7` | `baseline` | Model variant |
| `--mode` | `hrl`, `flat` | `hrl` | Training mode (overridden by `--variant`) |
| **Training** |
| `--episodes` | `N` | `500` | Number of episodes |
| `--turns` | `N` | `40` | Max turns per episode |
| `--lr` | `FLOAT` | `0.0001` | Learning rate |
| `--gamma` | `FLOAT` | `0.99` | Discount factor |
| `--device` | `cpu`, `cuda` | `cpu` | Device |
| **Experiment** |
| `--name` | `STRING` | auto-set | Experiment name (auto-set from variant) |
| `--experiment-type` | `major`, `minor` | `minor` | Experiment type (auto-set to `major` for variants) |
| **Rewards** |
| `--w-engagement` | `FLOAT` | `1.0` | Engagement reward scale |
| `--novelty-per-fact` | `FLOAT` | `1.0` | Novelty per new fact |
| `--w-responsiveness` | `FLOAT` | `0.25` | Responsiveness scale |
| `--w-conclude` | `FLOAT` | `0.5` | Conclude bonus per exhibit |
| `--w-transition-insufficiency` | `FLOAT` | `-0.20` | Transition penalty |
| **Visualization** |
| `--map-interval` | `N` | `50` | Map every N episodes (0=disable) |
| `--save-map-frames` | flag | - | Save map for every turn |
| **Debugging** |
| `--verbose` | flag | - | Show detailed output |
| `--show-prompts` | flag | - | Show LLM prompts |
| `--force-option` | `Explain`, `AskQuestion`, etc. | - | Force option (testing) |

### Variant Descriptions

- **`baseline`** - Hierarchical Option-Critic with Standard BERT
- **`h1`** - Flat Actor-Critic (no hierarchical structure)
- **`h3`** - Minimal Prompts (no structured headers)
- **`h5`** - State Ablation (dialogue-act-only state)
- **`h6`** - No Transition Rewards
- **`h7`** - Hybrid BERT (standard for intent, DialogueBERT for context)

---

## train_all_variations.py

**Train multiple model variations sequentially with identical parameters.**

### Quick Examples

```bash
# Train all variations
python train_all_variations.py --episodes 500 --device cuda --no-confirm

# Train specific variations
python train_all_variations.py --variations baseline h1 --episodes 500 --device cuda --no-confirm
```

### Arguments

| Argument | Options | Default | Description |
|----------|---------|---------|-------------|
| **Selection** |
| `--variations` | `baseline`, `h1`, `h3`, `h5`, `h6`, `h7`, `all` | `all` | Variations to train (can specify multiple) |
| **Training** |
| `--episodes` | `N` | `500` | Number of episodes |
| `--turns` | `N` | `40` | Max turns per episode |
| `--lr` | `FLOAT` | `0.0001` | Learning rate |
| `--gamma` | `FLOAT` | `0.99` | Discount factor |
| `--device` | `cpu`, `cuda` | `cuda` | Device |
| **Rewards** |
| `--w-engagement` | `FLOAT` | `1.0` | Engagement reward scale |
| `--novelty-per-fact` | `FLOAT` | `0.25` | Novelty per new fact |
| `--w-responsiveness` | `FLOAT` | `0.25` | Responsiveness scale |
| `--w-conclude` | `FLOAT` | `0.2` | Conclude bonus per exhibit |
| `--w-transition-insufficiency` | `FLOAT` | `-0.20` | Transition penalty |
| **Visualization** |
| `--map-interval` | `N` | `10` | Map every N episodes (0=disable) |
| `--save-map-frames` | flag | - | Save map for every turn |
| **Other** |
| `--no-confirm` | flag | - | Skip confirmation (recommended) |
| `--test-mode` | flag | - | Test mode (reduces output) |
| `--verbose` | flag | - | Verbose output |

### More Examples

```bash
# Overnight run (all variations)
python train_all_variations.py --episodes 500 --device cuda --no-confirm

# Quick test (5 episodes)
python train_all_variations.py --episodes 5 --map-interval 2 --test-mode --no-confirm

# Custom parameters
python train_all_variations.py --variations h3 h5 h7 --episodes 500 --device cuda --novelty-per-fact 0.3 --no-confirm
```

### Output

Results automatically saved to `major_results/YYYYMMDD_NNN_modelname/` after each training completes.

