# How To Analyze Results

## Step 1: Run Analysis

```bash
python create_evaluation_plots.py 7  # Replace 7 with your experiment number
```

Find your experiment number by checking `training_logs/experiments/`

## Step 2: Check Results

Open: `training_logs/experiments/exp_XXX/evaluation/`

You'll see 5 folders:

### 01_learning_dynamics/
- **learning_curve.png** - Is the agent learning? (should go up)
- **episode_length.png** - How long are conversations?

### 02_hierarchical_control/
- **option_distribution.png** - Which strategies does it use?
- **option_persistence.png** - Does it stick with strategies? (should be >2 turns)
- **option_evolution.png** - How does strategy change over training?

### 03_engagement_adaptation/
- **dwell_over_training.png** - Is attention improving? (should be >0.4)

### 04_content_quality/
- **exhibit_coverage.png** - Exploring museum? (should increase)
- **facts_presented.png** - Informative? (should be 5-10 per episode)

### 05_reward_decomposition/
- **reward_decomposition.png** - What drives learning?
- **reward_evolution.png** - How do rewards change?

## What "Good" Looks Like

| Metric | Good Value |
|--------|-----------|
| Learning curve | Going up |
| Option persistence | Mean > 2 turns |
| Dwell time | > 0.4 |
| Coverage | Increasing |
| Facts | 5-10 per episode |

## For Your Thesis

Use these plots to answer your research questions:
- **RQ1** (hierarchy): `option_persistence.png`, `option_evolution.png`
- **RQ2** (engagement): `dwell_over_training.png`
- **RQ3** (content): `exhibit_coverage.png`, `facts_presented.png`

That's it!

