# Evaluation Guide

## Automatic Evaluation

Evaluation runs **automatically** after every training session. No manual steps required!

Results are saved to: `training_logs/experiments/exp_XXX/evaluation/`

## What Gets Generated

### 1. Learning Dynamics (`01_learning_dynamics/`)
- **learning_curve.png** - Episode reward over time (should increase)
- **episode_length.png** - Conversation length distribution

### 2. Hierarchical Control (`02_hierarchical_control/`)
- **option_distribution.png** - Which strategies does the agent use?
- **option_persistence.png** - Does it stick with strategies? (should be >2 turns)
- **option_evolution.png** - How strategy usage changes over training

### 3. Engagement Adaptation (`03_engagement_adaptation/`)
- **dwell_over_training.png** - Attention improvement (should be >0.4)

### 4. Content Quality (`04_content_quality/`)
- **exhibit_coverage.png** - Museum exploration (should increase)
- **facts_presented.png** - Information delivery (should be 5-10 per episode)

### 5. Reward Decomposition (`05_reward_decomposition/`)
- **reward_decomposition.png** - What drives learning?
- **reward_evolution.png** - How rewards change over training

### 6. Summary
- **EVALUATION_SUMMARY.txt** - Human-readable summary

## Manual Re-run

If you need to regenerate evaluation plots:

```bash
python create_evaluation_plots.py <experiment_number>
```

Example:
```bash
python create_evaluation_plots.py 7
```

## Parameterization Analysis

After training, analyze reward weights:

```bash
python analyze_parameterization.py <experiment_number>
```

This generates recommendations for tuning reward weights based on actual training behavior.

Results saved to: `training_logs/experiments/exp_XXX/parameterization_results/`

## What "Good" Looks Like

| Metric | Good Value | Interpretation |
|--------|-----------|----------------|
| Learning curve | Going up | Agent is learning |
| Option persistence | Mean > 2 turns | Agent uses strategies properly |
| Dwell time | > 0.4 | Good engagement |
| Coverage | Increasing | Exploring museum |
| Facts | 5-10 per episode | Informative dialogue |

## For Your Thesis

Use these plots to answer research questions:

- **RQ1** (hierarchy): `option_persistence.png`, `option_evolution.png`
- **RQ2** (engagement): `dwell_over_training.png`
- **RQ3** (content): `exhibit_coverage.png`, `facts_presented.png`

