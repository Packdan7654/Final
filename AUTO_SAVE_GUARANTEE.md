# Automatic Saving Guarantee

## What is ALWAYS Saved (No Flags Needed)

Every training run automatically saves:

### 1. **Experiment Directory Structure**
- Created automatically: `training_logs/experiments/exp_XXX_NAME_TIMESTAMP/`
- Subdirectories: `logs/`, `maps/`, `metrics/`, `plots/`, `models/`, `checkpoints/`

### 2. **Metrics (ALWAYS SAVED)**
- `logs/metrics_tracker_TIMESTAMP.json` - Full metrics data
- `logs/monitor_TIMESTAMP_episodes.json` - Episode-level data
- `logs/monitor_TIMESTAMP_turns.json` - Turn-level data
- `logs/training_log_TIMESTAMP.json` - Complete training history

### 3. **Checkpoints (ALWAYS SAVED)**
- Every 50 episodes: `checkpoints/checkpoint_ep50_metrics.json`
- Every 50 episodes: `checkpoints/checkpoint_ep50_model.pt`
- Saves both metrics and model state

### 4. **Maps (ALWAYS SAVED)**
- `maps/episode_XXX_animation.gif` - Every N episodes (default: 50)
- Map visualizations showing agent/visitor navigation

### 5. **Models (ALWAYS SAVED)**
- `models/trained_agent.pt` - Final model checkpoint
- `models/final_model_epXXX.pt` - Final model with episode number
- All checkpoints saved to `checkpoints/`

### 6. **Evaluation Plots (ALWAYS GENERATED)**
- `evaluation/01_learning_dynamics/` - Learning curves, episode lengths
- `evaluation/02_hierarchical_control/` - Option usage, persistence, evolution
- `evaluation/03_engagement_adaptation/` - Dwell time analysis
- `evaluation/04_content_quality/` - Coverage, facts presented
- `evaluation/05_reward_decomposition/` - Reward component analysis
- `evaluation/EVALUATION_SUMMARY.txt` - Text summary

### 7. **Comprehensive Analysis (ALWAYS GENERATED)**
- `plots/learning_curve_TIMESTAMP.png`
- `plots/option_distribution_TIMESTAMP.png`
- `plots/option_evolution_TIMESTAMP.png`
- `plots/reward_distribution_TIMESTAMP.png`
- `plots/comprehensive_analysis_TIMESTAMP.png`

### 8. **Metadata (ALWAYS SAVED)**
- `metadata.json` - Experiment configuration
- `summary.json` - Final training summary

## No Configuration Needed

Just run:
```bash
python train.py --episodes 1000 --turns 40 --map-interval 50
```

Everything is saved automatically. No flags needed.

## What Changed

1. ✅ `save_metrics` defaults to `True` (forced to True in train.py)
2. ✅ `enable_map_viz` defaults to `True` (forced to True in train.py)
3. ✅ `map_interval` defaults to `50` (was 5)
4. ✅ Experiment directory created automatically
5. ✅ All saves go to experiment directory
6. ✅ Checkpoints always save (not conditional)
7. ✅ Metrics always save (not conditional)
8. ✅ Evaluation plots generated automatically
9. ✅ Comprehensive analysis enabled automatically

## Experiment Structure

```
training_logs/experiments/exp_XXX_NAME_TIMESTAMP/
├── logs/
│   ├── metrics_tracker_*.json
│   ├── monitor_*_episodes.json
│   ├── monitor_*_turns.json
│   └── training_log_*.json
├── maps/
│   └── episode_XXX_animation.gif
├── models/
│   ├── trained_agent.pt
│   └── final_model_epXXX.pt
├── checkpoints/
│   ├── checkpoint_ep50_metrics.json
│   ├── checkpoint_ep50_model.pt
│   └── ...
├── plots/
│   └── *.png (comprehensive analysis)
├── evaluation/
│   ├── 01_learning_dynamics/
│   ├── 02_hierarchical_control/
│   ├── 03_engagement_adaptation/
│   ├── 04_content_quality/
│   ├── 05_reward_decomposition/
│   └── EVALUATION_SUMMARY.txt
├── metadata.json
└── summary.json
```

