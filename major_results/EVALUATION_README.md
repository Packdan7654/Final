# Evaluation Guide

This guide explains how to run evaluations and where results are automatically saved.

## Automatic Evaluation

**Evaluation runs automatically after training completes.** Results are saved to `evaluation/` in the experiment directory.

## Manual Evaluation

### Evaluate Specific Experiment

```bash
# H1: Option Structure
python experiments/h1_option_structure/evaluate.py \
    --experiment-dir major_results/20251112_001_h1_flat_policy \
    --output-dir major_results/20251112_001_h1_flat_policy/evaluation

# H3: Minimal Prompts
python experiments/h3_prompt_headers/evaluate.py \
    --experiment-dir major_results/20251112_001_h3_minimal_prompts \
    --output-dir major_results/20251112_001_h3_minimal_prompts/evaluation

# H5: State Ablation
python experiments/h5_state_ablation/evaluate.py \
    --experiment-dir major_results/20251112_001_h5_state_ablation \
    --output-dir major_results/20251112_001_h5_state_ablation/evaluation

# H6: Transition Reward
python experiments/h6_transition_reward/evaluate.py \
    --experiment-dir major_results/20251112_001_h6_transition_reward \
    --output-dir major_results/20251112_001_h6_transition_reward/evaluation

# H7: Hybrid BERT
python experiments/h7_hybrid_bert/evaluate.py \
    --experiment-dir major_results/20251112_001_h7_hybrid_bert \
    --output-dir major_results/20251112_001_h7_hybrid_bert/evaluation
```

### Evaluate All Experiments

```bash
# Run all evaluations
python experiments/run_all_evaluations.py
```

### Evaluate with Custom Episodes

```bash
# H3: Evaluate with 500 episodes
python experiments/h3_prompt_headers/evaluate_with_prompts.py \
    --num-episodes 500 \
    --experiment-dir major_results/20251112_001_h3_minimal_prompts
```

## Evaluation Output

### Location

Evaluation metrics are saved to:
```
major_results/YYYYMMDD_NNN_modelname/evaluation/
```

### Files Generated

Each evaluation creates hypothesis-specific metrics files:

- **H1**: `h1_metrics.json` - Option coherence, switch rates, persistence
- **H2**: `h2_metrics.json` - Termination function analysis
- **H3**: `h3_metrics.json` - Prompt effectiveness metrics
- **H5**: `h5_metrics.json` - State ablation comparison
- **H6**: `h6_metrics.json` - Transition reward impact
- **H7**: `h7_metrics.json` - BERT variant comparison

### Example Metrics Structure

```json
{
  "mean_return": 45.2,
  "std_return": 12.3,
  "mean_episode_length": 28.5,
  "coherent_span_lengths": [5, 8, 3, ...],
  "switch_rate_per_100_turns": 12.5,
  ...
}
```

## Evaluation Plots

Evaluation-specific plots are generated in:
```
major_results/YYYYMMDD_NNN_modelname/visualizations/evaluation/
```

### Generate Evaluation Plots

```bash
# Generate plots for specific experiment
python tools/generate_major_results_visualizations.py \
    --experiment 20251112_001_baseline \
    --advanced

# Generate for all experiments
python tools/generate_major_results_visualizations.py \
    --all \
    --advanced
```

## Comparing Models

### Compare Specific Models

```python
from experiments.shared.comparison_tools import compare_models

# Compare baseline vs H1
compare_models(
    'major_results/20251112_001_baseline',
    'major_results/20251112_001_h1_flat_policy'
)
```

### Command Line Comparison

```bash
# Use comparison tools
python experiments/shared/comparison_tools.py \
    --model1 major_results/20251112_001_baseline \
    --model2 major_results/20251112_001_h1_flat_policy
```

## Evaluation Metrics by Hypothesis

### H1: Option Structure
- **Coherent span lengths**: Consecutive turns under same option
- **Switch rate**: Switches per 100 turns
- **Option persistence**: Average duration per option

### H2: Learned Terminations
- **Termination frequency**: How often options terminate
- **Termination accuracy**: Correlation with optimal termination points

### H3: Minimal Prompts
- **Prompt effectiveness**: Comparison of structured vs minimal prompts
- **Response quality**: Metrics on generated responses

### H5: State Ablation
- **State component importance**: Impact of removing state components
- **Performance comparison**: Ablation vs full state

### H6: Transition Reward
- **Transition behavior**: Impact of removing transition rewards
- **Strategy switching**: How transition rewards affect option selection

### H7: Hybrid BERT
- **BERT variant comparison**: Standard vs DialogueBERT vs Hybrid
- **Embedding quality**: Impact on state representation

## Troubleshooting

### Evaluation not running automatically
- Check that training completed successfully
- Verify `experiment-type major` was used
- Run evaluation manually using commands above

### Missing evaluation metrics
- Check `evaluation/` directory exists
- Verify evaluation script path is correct
- Check console output for errors

### Evaluation takes too long
- Reduce `--num-episodes` for faster evaluation
- Use `--device cuda` if available
- Check for performance bottlenecks in evaluation script

## Next Steps

After evaluation:
1. Review metrics in `evaluation/*_metrics.json`
2. Generate evaluation plots (see [VISUALIZATION_README.md](VISUALIZATION_README.md))
3. Compare with other models
4. Analyze results for thesis

