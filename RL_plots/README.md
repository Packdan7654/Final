# RL/HRL Evaluation Plots

This directory contains individual plot generators for comprehensive RL/HRL evaluation.

## Generated Plots

### 1. Learning Curves
- **01_learning_curve.png**: Episode returns over training with moving average and confidence intervals
- **02_episode_length.png**: Episode duration evolution (sample efficiency)

### 2. Value Function Analysis
- **03_value_function.png**: Critic value estimates over training

### 3. Policy Gradient Metrics
- **04_policy_loss.png**: Actor (policy) loss evolution
- **05_value_loss.png**: Critic (value) loss evolution
- **06_policy_entropy.png**: Exploration vs exploitation trade-off

### 4. Advantage Analysis
- **07_advantage_distribution.png**: Histogram of advantage values

### 5. HRL-Specific Analysis
- **08_option_transitions.png**: Option transition matrix (counts and probabilities)
- **09_option_durations.png**: Option persistence duration distribution

### 6. Reward Analysis
- **10_reward_distribution.png**: Histogram of episode rewards
- **11_reward_decomposition.png**: Reward component contributions over time

### 7. Training Stability
- **12_training_stability.png**: Rolling mean, variance, and coefficient of variation

### 8. Policy Analysis
- **13_action_distribution.png**: Action (option) usage frequency

## Usage

### Automatic Generation (Recommended)
**RL plots are automatically generated after training completes!** The plots are saved to `<experiment_dir>/RL_plots/` automatically.

### Manual Generation
If you need to regenerate plots manually:

```bash
# Generate all plots (saves to <experiment_path>/RL_plots by default)
python RL_plots/generate_rl_plots.py training_logs/experiments/20251105/exp_003_20251105_161942

# Custom output directory:
python RL_plots/generate_rl_plots.py <experiment_path> --output my_plots
```

### Example:
```bash
python RL_plots/generate_rl_plots.py training_logs/experiments/20251105/exp_003_20251105_161942
```

**Note:** By default (when `--output` is not specified), plots are saved to `<experiment_path>/RL_plots/`, making them easy to find alongside other experiment outputs.

## Requirements

- matplotlib
- seaborn
- numpy
- Training data in JSON format (from metrics_tracker)

## Plot Characteristics

All plots are generated with:
- 300 DPI resolution (publication quality)
- Whitegrid style for clarity
- Consistent color schemes
- Clear labels and legends
- Statistical annotations where relevant

## Integration with Existing Evaluation

These plots complement the existing evaluation plots in `create_evaluation_plots.py` by focusing specifically on RL/HRL metrics (value functions, policy gradients, advantages, etc.) rather than domain-specific metrics (coverage, engagement, etc.).
