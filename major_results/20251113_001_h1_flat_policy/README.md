# H1: Flat Actor-Critic (No Hierarchical Structure)

## Model Description

This variant tests the hypothesis that hierarchical option structure improves long-horizon behavior. It uses a flat Actor-Critic policy over all primitive actions (12 actions = 4 options × 3 subactions) without hierarchical structure.

### Architecture
- **Algorithm**: Standard Actor-Critic with TD(0) learning
- **State Representation**: Same as baseline (149-d)
- **BERT Mode**: Standard BERT (same as baseline)
- **Action Space**: Flat discrete space (12 actions)
- **No Options**: Direct policy over primitive actions

### Key Differences from Baseline
- **No hierarchical structure**: Single policy head instead of option-level + intra-option policies
- **No termination functions**: Actions are selected directly, no option duration learning
- **Same state representation**: Uses same 149-d state as baseline
- **Same rewards**: Identical reward function

### Hypothesis (H1)
An option-based manager will outperform a flat policy on long-horizon objectives. We expect higher episodic return with lower variance, longer coherent stretches under a chosen strategy, and fewer needless switches between strategies.

### Training Configuration
- **Episodes**: 500
- **Learning Rate**: 0.0001
- **Gamma**: 0.99
- **Device**: cuda
- **Training Date**: 2025-11-13T00:18:19.619141

## Results Location

- **Training Results**: `training/`
- **Evaluation Metrics**: `evaluation/`
- **Visualizations**: `visualizations/`
- **Consolidated Metrics**: `metrics/`

## Key Metrics

See `metrics/training_metrics.json` and `evaluation/h1_metrics.json` for:
- Episode returns (compared to baseline)
- Coherent span lengths (consecutive turns under same strategy)
- Switch rate (switches per 100 turns)
- Option duration statistics (not applicable - flat policy)
- Policy entropy
