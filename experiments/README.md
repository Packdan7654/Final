# Experimental Framework for Hypothesis Testing

This directory contains experimental variants for testing all hypotheses (H1-H6) from `paper.tex`.

**📖 Documentation:**
- **[TESTING_TIMELINE.md](TESTING_TIMELINE.md)** - ⭐ **START HERE** - Complete step-by-step testing guide with timeline
- **[HYPOTHESIS_IMPLEMENTATION.md](HYPOTHESIS_IMPLEMENTATION.md)** - Detailed implementation guide for all hypotheses
- **[BASELINE_JUSTIFICATION.md](BASELINE_JUSTIFICATION.md)** - Justification for flat RL baseline architecture (H1)

## Structure

```
experiments/
├── shared/                    # Shared utilities
│   ├── dialogue_act_classifier.py    # Dialogue act classification for H5
│   ├── evaluation_framework.py       # Standardized evaluation metrics
│   └── comparison_tools.py           # Cross-variant comparison tools
│
├── h1_option_structure/        # H1: Flat policy vs options
│   ├── train.py                # Training script
│   └── evaluate.py             # Evaluation script
│
├── h2_learned_terminations/    # H2: Fixed duration vs learned terminations
│   ├── env.py                  # Environment with fixed durations
│   ├── train.py                # Training script
│   └── evaluate.py             # Evaluation script
│
├── h3_prompt_headers/          # H3: Minimal prompts vs structured headers
│   ├── train.py                # Training script
│   └── evaluate.py             # Evaluation script
│
├── h5_state_ablation/          # H5: State ablation experiment
│   ├── env.py                  # Environment with dialogue-act state
│   ├── train.py                # Training script
│   └── evaluate.py             # Evaluation script
│
└── h6_transition_reward/       # H6: Transition reward experiment
    ├── env.py                  # Environment with reward toggle
    ├── train.py                # Training script
    └── evaluate.py             # Evaluation script
```

## Hypotheses

### H1: Option Structure vs Flat Policy
**Question**: Does an option-based manager outperform a flat policy on long-horizon objectives?

**Variant**: Flat RL system with primitive actions only (no hierarchical options).

**Metrics**: Coherent span lengths, switch rates, option persistence, episodic return.

### H2: Learned Terminations
**Question**: Do learned terminations for Explain track engagement and intent better than fixed durations?

**Variant**: Fixed-duration Explain option (3 turns) vs learned termination functions.

**Metrics**: Explain duration, dwell correlation, termination timing.

### H3: Prompt Headers
**Question**: Do slot-filled prompt headers improve realization quality (faithfulness, KB grounding, less repetition)?

**Variant**: Minimal prompts without structured headers vs full structured headers.

**Metrics**: Novel fact coverage, repetition ratio, KB citation rate, hallucination rate.

### H5: State Ablation
**Question**: Can a compact dialogue-act state representation match or exceed performance of full DialogueBERT embeddings?

**Variant**: Replaces DialogueBERT embeddings (128-d) with dialogue act classification (6-d one-hot).

**State Reduction**: 149-d → ~20-d (for 5 exhibits)

### H6: Transition Reward Shaping
**Question**: Does transition probability-based reward shaping improve transition timing and overall performance?

**Variant**: Removes explicit transition insufficiency/sufficiency rewards, keeping only probabilistic transition acceptance.

## Usage

### 1. Train Baseline (Reference)

First, train the baseline system for comparison:

```bash
python train.py --episodes 600 --name baseline --device cuda
```

This creates: `training_logs/experiments/YYYYMMDD/exp_XXX_baseline_*/`

### 2. Train All Variants (Recommended)

Use the master script to run all experiments:

```bash
python experiments/run_all_experiments.py \
    --episodes 600 \
    --device cuda \
    --skip-baseline  # Skip if already trained
```

Or train individual variants:

#### H1: Flat Policy
```bash
python experiments/h1_option_structure/train.py \
    --episodes 600 \
    --name h1_flat_policy \
    --device cuda
```

#### H2: Fixed Duration
```bash
python experiments/h2_learned_terminations/train.py \
    --episodes 600 \
    --name h2_fixed_duration \
    --device cuda
```

#### H3: Minimal Prompts
```bash
python experiments/h3_prompt_headers/train.py \
    --episodes 600 \
    --name h3_minimal_prompts \
    --device cuda
```

#### H5: State Ablation
```bash
python experiments/h5_state_ablation/train.py \
    --episodes 600 \
    --name h5_state_ablation \
    --device cuda
```

#### H6: Transition Reward
```bash
python experiments/h6_transition_reward/train.py \
    --episodes 600 \
    --name h6_transition_reward \
    --device cuda
```

### 3. Evaluate Experiments

Each hypothesis has its own evaluation script:

#### H1: Option Structure
```bash
python experiments/h1_option_structure/evaluate.py \
    --experiment-dir training_logs/experiments/YYYYMMDD/major_XXX_h1_flat_policy_* \
    --output-dir experiments/results/h1
```

#### H2: Learned Terminations
```bash
python experiments/h2_learned_terminations/evaluate.py \
    --experiment-dir training_logs/experiments/YYYYMMDD/major_XXX_h2_fixed_duration_* \
    --output-dir experiments/results/h2
```

#### H3: Prompt Headers
```bash
python experiments/h3_prompt_headers/evaluate.py \
    --experiment-dir training_logs/experiments/YYYYMMDD/major_XXX_h3_minimal_prompts_* \
    --output-dir experiments/results/h3
```

#### H5: State Ablation
```bash
python experiments/h5_state_ablation/evaluate.py \
    --experiment-dir training_logs/experiments/YYYYMMDD/major_XXX_h5_state_ablation_* \
    --output-dir experiments/results/h5
```

#### H6: Transition Reward
```bash
python experiments/h6_transition_reward/evaluate.py \
    --experiment-dir training_logs/experiments/YYYYMMDD/major_XXX_h6_transition_reward_* \
    --output-dir experiments/results/h6
```

#### H4: Training Stability
H4 compares training dynamics between baseline (hierarchical) and H1 (flat):
```bash
python experiments/h4_training_stability/evaluate.py \
    --hierarchical-dir training_logs/experiments/YYYYMMDD/exp_XXX_baseline_* \
    --flat-dir training_logs/experiments/YYYYMMDD/major_XXX_h1_flat_policy_* \
    --output-dir experiments/results/h4
```

### 5. Compare Results

```python
from experiments.shared.comparison_tools import HypothesisComparator
from pathlib import Path

comparator = HypothesisComparator(Path('experiments/results'))
report = comparator.generate_comparison_report(
    Path('experiments/results/comparison_report.json')
)
```

This generates:
- `comparison_report.json`: Detailed comparison metrics
- `comparison_report.txt`: Human-readable summary

## Metrics

### H1 Metrics (Option Structure)
- **Coherent Span Lengths**: Consecutive turns under same option
- **Switch Rate**: Option switches per 100 turns
- **Option Persistence**: Average duration per option
- **Performance Comparison**: Return, coverage vs baseline

### H2 Metrics (Learned Terminations)
- **Explain Durations**: Duration of Explain option segments
- **Dwell Correlation**: Correlation between dwell and Explain duration
- **Termination Timing**: Timing of terminations relative to intent changes
- **Performance Comparison**: Return vs baseline

### H3 Metrics (Prompt Headers)
- **Novel Fact Coverage**: Coverage of novel facts
- **Repetition Ratio**: Ratio of repeated facts
- **KB Citation Rate**: Rate of KB citations
- **Hallucination Rate**: Rate of non-KB claims
- **Performance Comparison**: Return vs baseline

### H5 Metrics (State Ablation)
- **State Dimension**: Reduced state vector size
- **Compression Ratio**: Reduction vs baseline (149-d)
- **Dialogue Act Distribution**: Frequency of each act type
- **Performance Comparison**: Return, coverage vs baseline

### H6 Metrics (Transition Reward)
- **Transition Success Rate**: Success rate with/without reward shaping
- **Transition Timing**: Average facts before transition
- **Reward Component Breakdown**: Contribution of each reward component
- **Performance Comparison**: Return, coverage vs baseline

### H4 Metrics (Training Stability)
- **Learning Curve Smoothness**: Variance of return differences
- **Update Magnitude**: Average policy update size
- **Time to Target**: Episodes to reach 80% of max return
- **Comparison**: Hierarchical vs flat training dynamics

### Common Metrics (All Experiments)
- Mean return and standard deviation
- Mean episode length
- Mean coverage ratio
- Total episodes trained

## Output Structure

Each experiment generates:

```
training_logs/experiments/YYYYMMDD/major_XXX_<variant>_*/
├── metadata.json              # Experiment configuration
├── logs/
│   ├── metrics_tracker_*.json  # Training metrics
│   └── monitor_*.json         # Turn/episode data
├── models/
│   └── trained_agent.pt       # Saved model
└── evaluation/                # (after running evaluate.py)
    └── <variant>_metrics.json # Hypothesis-specific metrics
```

## Comparison Workflow

1. **Train all variants** with same hyperparameters (episodes, turns, lr, gamma)
2. **Evaluate each variant** using respective evaluation scripts
3. **Generate comparison report** using comparison tools
4. **Analyze results** in `experiments/results/comparison_report.json`

## Notes

- All experiments use the **same simulator** and **same reward structure** (except H6's transition reward toggle)
- Training hyperparameters should be **identical** across variants for fair comparison
- Results are saved with experiment metadata for reproducibility
- Evaluation scripts automatically extract hypothesis-specific metrics

## Extending the Framework

To add a new hypothesis variant:

1. Create new directory: `experiments/hX_new_hypothesis/`
2. Create `env.py` extending base environment
3. Create `train.py` with variant-specific training loop
4. Create `evaluate.py` with hypothesis-specific metrics
5. Add metrics computation to `shared/evaluation_framework.py`
6. Add comparison logic to `shared/comparison_tools.py`

