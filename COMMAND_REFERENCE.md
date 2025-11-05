# Complete Command Reference

## Basic Training

### Minimal Command
```bash
python train.py --episodes 600 --device cuda
```
**Output:**
- Saves map animations every 50 episodes (default)
- Runs evaluation automatically
- Saves all metrics and logs

### With Experiment Name
```bash
python train.py --episodes 600 --device cuda --name "baseline"
```
**Output:**
- Creates folder: `training_logs/experiments/exp_001_baseline_TIMESTAMP/`
- All outputs saved there

---

## Reward Weight Configuration

### Default Weights (Paper Baseline)
```bash
python train.py --episodes 600 --device cuda
```
**Reward weights:**
- `w_engagement = 1.0`
- `w_novelty = 0.5`
- `w_responsiveness = 0.25`
- `w_conclude = 0.2`

**Note:** Penalties (question spam, early transitions) are handled at simulator level via reduced dwell time. Bad agent choices naturally reduce engagement.

### Increase Novelty (Encourage More Facts)
```bash
python train.py --episodes 600 --w-novelty 1.0 --device cuda
```
**What changes:**
- Novelty reward weight: 0.5 → 1.0 (doubled)
- Agent will prioritize sharing new facts more

### Balance Engagement and Novelty
```bash
python train.py --episodes 600 --w-engagement 1.0 --w-novelty 1.0 --device cuda
```
**What changes:**
- Equal weights for engagement and novelty
- Balanced learning between attention and content

### Reduce Question Spam
**Note:** Question spam penalties are now handled automatically by the simulator. When the agent asks too many questions in a row, the simulator reduces dwell time (lower engagement), which naturally reduces the engagement reward. No command-line parameter needed - it's automatic!

### Custom Novelty Parameters
```bash
python train.py --episodes 600 --novelty-per-fact 0.2 --novelty-explain-bonus 0.15 --device cuda
```
**What changes:**
- More reward per fact: 0.15 → 0.2
- Bigger bonus for ExplainNewFact: 0.1 → 0.15

---

## Map Visualization

### Default (Every 50 Episodes)
```bash
python train.py --episodes 600 --device cuda
```
**Output:**
- `training_logs/experiments/exp_XXX/maps/episode_000_animation.gif`
- `training_logs/experiments/exp_XXX/maps/episode_050_animation.gif`
- `training_logs/experiments/exp_XXX/maps/episode_100_animation.gif`
- ... (every 50 episodes)

### Custom Interval
```bash
python train.py --episodes 600 --map-interval 10 --device cuda
```
**Output:**
- Maps saved every 10 episodes instead of 50
- More frequent visualizations

### Save Every Turn (All Episodes)
```bash
python train.py --episodes 600 --save-map-frames --device cuda
```
**Output:**
- `training_logs/experiments/exp_XXX/maps/episode_000/turn_01.png`
- `training_logs/experiments/exp_XXX/maps/episode_000/turn_02.png`
- `training_logs/experiments/exp_XXX/maps/episode_000/turn_03.png`
- ... (every turn of every episode)
- **WARNING:** Creates many files! Use sparingly.

### Disable Maps
```bash
python train.py --episodes 600 --map-interval 0 --device cuda
```
**Output:**
- No map visualizations saved

---

## Training Configuration

### Short Test Run
```bash
python train.py --episodes 3 --turns 10 --name "test" --device cuda
```
**Output:**
- 3 episodes, 10 turns each
- Quick validation (~1 minute)

### Long Training Run
```bash
python train.py --episodes 1000 --turns 40 --name "long_run" --device cuda
```
**Output:**
- 1000 episodes for extended training

### Custom Learning Rate
```bash
python train.py --episodes 600 --lr 1e-4 --device cuda
```
**What changes:**
- Learning rate: 3e-4 → 1e-4 (slower learning)

---

## Debugging & Testing

### Show Prompts
```bash
python train.py --episodes 10 --show-prompts --device cuda
```
**Output:**
- Prints all LLM prompts to console
- Useful for debugging prompt generation

### Force Option (Testing)
```bash
python train.py --episodes 10 --force-option Explain --device cuda
```
**What happens:**
- Agent always chooses Explain option
- Ignores policy, forces specific behavior

### Verbose Output
```bash
python train.py --episodes 10 --verbose --device cuda
```
**Output:**
- Detailed turn-by-turn information
- Reward breakdowns each turn

### Live Monitor
```bash
python train.py --episodes 10 --enable-live-monitor --device cuda
```
**Output:**
- Real-time visualization windows
- Interactive training monitor

---

## Complete Output Structure

Every training run creates:

```
training_logs/experiments/exp_XXX_NAME_TIMESTAMP/
├── metadata.json              # Experiment configuration
├── summary.json               # Final summary
│
├── models/
│   └── trained_agent.pt       # Saved model
│
├── logs/
│   ├── training_metrics.json  # All metrics
│   └── episode_logs.json      # Episode summaries
│
├── maps/
│   ├── episode_000_animation.gif  # Map animations (if --map-interval > 0)
│   ├── episode_050_animation.gif
│   └── episode_XXX/turn_YY.png    # Individual frames (if --save-map-frames)
│
├── detailed_logs/
│   ├── episode_00000/
│   │   ├── episode_log.json       # Full episode data
│   │   └── dialogue_summary.txt   # Human-readable summary
│   └── episode_XXXXX/...
│
├── parameterization_results/
│   ├── parameterization_report.json
│   └── parameterization_summary.txt
│
└── evaluation/                # ALWAYS GENERATED
    ├── 01_learning_dynamics/
    │   ├── learning_curve.png
    │   └── episode_length.png
    ├── 02_hierarchical_control/
    │   ├── option_distribution.png
    │   ├── option_persistence.png
    │   └── option_evolution.png
    ├── 03_engagement_adaptation/
    │   └── dwell_over_training.png
    ├── 04_content_quality/
    │   ├── exhibit_coverage.png
    │   └── facts_presented.png
    ├── 05_reward_decomposition/
    │   ├── reward_decomposition.png
    │   └── reward_evolution.png
    └── EVALUATION_SUMMARY.txt
```

---

## Common Workflows

### 1. Quick Test
```bash
python train.py --episodes 3 --turns 10 --name "test" --device cuda
```

### 2. Baseline Training
```bash
python train.py --episodes 600 --name "baseline" --device cuda
```

### 3. Tune for More Facts
```bash
python train.py --episodes 600 --w-novelty 1.0 --novelty-per-fact 0.2 --name "high_novelty" --device cuda
```

### 4. Debug Specific Episode
```bash
python train.py --episodes 10 --verbose --show-prompts --name "debug" --device cuda
```

### 5. Full Training with All Metrics
```bash
python train.py --episodes 1000 --map-interval 25 --save-map-frames --name "full_run" --device cuda
```

---

## All Available Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--episodes` | int | 600 | Number of training episodes |
| `--turns` | int | 40 | Max turns per episode |
| `--lr` | float | 3e-4 | Learning rate |
| `--gamma` | float | 0.99 | Discount factor |
| `--device` | str | cpu | Device (cpu/cuda) |
| `--name` | str | None | Experiment name |
| `--w-engagement` | float | 1.0 | Engagement reward weight |
| `--w-novelty` | float | 0.5 | Novelty reward weight |
| `--w-responsiveness` | float | 0.25 | Responsiveness reward weight |
| `--w-conclude` | float | 0.2 | Conclude bonus weight |
| | | | **Note:** Penalties handled at simulator level (reduced dwell) |
| `--novelty-per-fact` | float | 0.15 | Reward per new fact |
| `--novelty-explain-bonus` | float | 0.1 | Bonus for ExplainNewFact |
| `--map-interval` | int | 50 | Save maps every N episodes (0=disable) |
| `--save-map-frames` | flag | False | Save map at EVERY turn |
| `--live-map-display` | flag | False | Show live map windows |
| `--show-prompts` | flag | False | Show LLM prompts |
| `--force-option` | str | None | Force specific option |
| `--force-subaction` | str | None | Force specific subaction |
| `--enable-live-monitor` | flag | False | Enable live monitor |
| `--verbose` | flag | False | Verbose output |

