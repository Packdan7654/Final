# Visualization Guide

This guide explains how to generate plots and visualizations for your experiments.

## Automatic Plot Generation

**Basic plots are automatically generated during training** and saved to:
```
major_results/YYYYMMDD_NNN_modelname/visualizations/basic/
```

### Auto-Generated Plots

1. **`learning_curve.png`**
   - Episode returns with moving average
   - Shows training progress over episodes

2. **`convergence_analysis.png`**
   - Convergence analysis with moving average
   - Marks convergence point if detected

3. **`rl_metrics_summary.png`**
   - 2x2 grid showing:
     - Value loss
     - Policy loss
     - Entropy
     - Value estimates

## Generating Advanced Plots

### For Latest Experiment of a Model

```bash
# Generate advanced plots for latest baseline
python tools/generate_major_results_visualizations.py \
    --model baseline \
    --advanced

# Generate for latest H1 experiment
python tools/generate_major_results_visualizations.py \
    --model h1_flat_policy \
    --advanced
```

### For Specific Experiment

```bash
# Generate advanced plots for specific experiment
python tools/generate_major_results_visualizations.py \
    --experiment 20251112_001_baseline \
    --advanced
```

### For All Experiments

```bash
# Generate advanced plots for all experiments
python tools/generate_major_results_visualizations.py \
    --all \
    --advanced
```

## Advanced Plot Types

Advanced plots are saved to:
```
major_results/YYYYMMDD_NNN_modelname/visualizations/advanced/
```

### Available Advanced Plots

1. **Option Usage Analysis**
   - Option selection frequency over time
   - Option duration distributions
   - Option switching patterns

2. **Reward Component Breakdown**
   - Engagement rewards over time
   - Novelty rewards over time
   - Responsiveness rewards over time
   - Transition rewards over time

3. **State Analysis**
   - State component distributions
   - State space coverage
   - State transition patterns

4. **Policy Analysis**
   - Policy entropy over time
   - Action distribution heatmaps
   - Value function estimates

5. **Convergence Analysis**
   - Detailed convergence metrics
   - Learning rate schedules
   - Gradient norms

## Evaluation Plots

Evaluation-specific plots are generated during evaluation and saved to:
```
major_results/YYYYMMDD_NNN_modelname/visualizations/evaluation/
```

### Generate Evaluation Plots

```bash
# Generate evaluation plots for specific experiment
python tools/generate_major_results_visualizations.py \
    --experiment 20251112_001_baseline \
    --advanced

# This will also generate evaluation plots if evaluation metrics exist
```

## Custom Plot Generation

### Using Python

```python
from src.visualization.major_results_plotter import MajorResultsPlotter
from pathlib import Path

# Load experiment directory
exp_dir = Path('major_results/20251112_001_baseline')

# Create plotter
plotter = MajorResultsPlotter(exp_dir)

# Generate all advanced plots
plotter.generate_all_advanced_plots()

# Generate specific plot
plotter.plot_option_usage()
plotter.plot_reward_breakdown()
plotter.plot_state_analysis()
```

## Plot Locations

### Directory Structure

```
major_results/YYYYMMDD_NNN_modelname/
└── visualizations/
    ├── basic/              # Auto-generated during training
    │   ├── learning_curve.png
    │   ├── convergence_analysis.png
    │   └── rl_metrics_summary.png
    ├── advanced/           # Generated on-demand
    │   ├── option_usage.png
    │   ├── reward_breakdown.png
    │   ├── state_analysis.png
    │   └── ...
    └── evaluation/         # Generated during evaluation
        ├── comparison_plots.png
        └── ...
```

## Viewing Plots

### Command Line

```bash
# View basic plots
ls major_results/20251112_001_baseline/visualizations/basic/

# View advanced plots
ls major_results/20251112_001_baseline/visualizations/advanced/

# View evaluation plots
ls major_results/20251112_001_baseline/visualizations/evaluation/
```

### Python

```python
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image

# Load and display plot
img_path = Path('major_results/20251112_001_baseline/visualizations/basic/learning_curve.png')
img = Image.open(img_path)
img.show()
```

## Troubleshooting

### Plots not generating
- Check that training completed successfully
- Verify metrics exist in `metrics/` directory
- Check console output for errors

### Missing advanced plots
- Advanced plots require running: `python tools/generate_major_results_visualizations.py --advanced`
- Basic plots are auto-generated during training

### Plot quality issues
- Plots are saved at 300 DPI by default
- Check matplotlib backend if plots fail to save
- Verify sufficient disk space

## Next Steps

After generating plots:
1. Review basic plots in `visualizations/basic/`
2. Generate advanced plots for detailed analysis
3. Compare plots across different models
4. Use plots for thesis figures

