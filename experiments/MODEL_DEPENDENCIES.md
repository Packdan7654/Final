# Model Dependencies and Training Requirements

This document clarifies which models need to be trained and which can reuse existing models to minimize training time.

## Training Requirements Summary

| Hypothesis | Needs Training? | Model Type | Compares To | Metrics Location |
|------------|----------------|------------|-------------|------------------|
| **Baseline** | ✅ YES | Hierarchical (Option-Critic) | N/A (baseline) | `training_logs/experiments/*/exp_*_baseline_*/` |
| **H1** | ✅ YES | Flat (Actor-Critic) | Baseline | `training_logs/experiments/*/major_*_h1_flat_policy_*/` |
| **H2** | ❌ NO | Uses Baseline Only | Baseline | N/A (evaluates baseline model) |
| **H3** | ✅ YES | Hierarchical (Minimal Prompts) | Baseline | `training_logs/experiments/*/major_*_h3_minimal_prompts_*/` |
| **H4** | ❌ NO | Uses Baseline + H1 | Baseline vs H1 | Compares existing models |
| **H5** | ✅ YES | Hierarchical (State Ablation) | Baseline | `training_logs/experiments/*/major_*_h5_state_ablation_*/` |
| **H6** | ✅ YES | Hierarchical (No Transition Rewards) | Baseline | `training_logs/experiments/*/major_*_h6_transition_reward_*/` |
| **H7** | ✅ YES | Hierarchical (Hybrid BERT) | Baseline | `training_logs/experiments/*/major_*_h7_hybrid_bert_*/` |

## Dependency Graph

```
Baseline (Hierarchical Option-Critic)
  ├─→ H1 (Flat Actor-Critic) ──┐
  ├─→ H2 (Fixed Duration)      │
  ├─→ H3 (Minimal Prompts)      │
  ├─→ H5 (State Ablation)       │
  ├─→ H6 (No Transition Rewards)│
  │                             │
  └─────────────────────────────┼─→ H4 (Training Stability)
                                 │   (Compares Baseline vs H1)
                                 │
                                 └─→ All comparisons use Baseline
```

## Training Order (Minimize Dependencies)

### Phase 1: Core Models (Required for all comparisons)
1. **Baseline** - Must train first (all other hypotheses compare to this)
2. **H1** - Required for H4 comparison

### Phase 2: Independent Variants (Can train in parallel)
3. **H2** - Uses Baseline Only (no training needed)
4. **H3** - Compares to Baseline  
5. **H5** - Compares to Baseline
6. **H6** - Compares to Baseline

### Phase 3: Analysis (No training needed)
7. **H4** - Uses Baseline + H1 (already trained)

## Model Comparison Matrix

| Hypothesis | Baseline Model | Variant Model | Comparison Metrics Location |
|------------|----------------|---------------|----------------------------|
| H1 | `exp_*_baseline_*` | `major_*_h1_flat_policy_*` | `experiments/results/h1/` |
| H2 | `exp_*_baseline_*` | `major_*_h2_fixed_duration_*` | `experiments/results/h2/` |
| H3 | `exp_*_baseline_*` | `major_*_h3_minimal_prompts_*` | `experiments/results/h3/` |
| H4 | `exp_*_baseline_*` + `major_*_h1_flat_policy_*` | N/A (compares two models) | `experiments/results/h4/` |
| H5 | `exp_*_baseline_*` | `major_*_h5_state_ablation_*` | `experiments/results/h5/` |
| H6 | `exp_*_baseline_*` | `major_*_h6_transition_reward_*` | `experiments/results/h6/` |
| H7 | `exp_*_baseline_*` | `major_*_h7_hybrid_bert_*` | `experiments/results/h7/` |

## Quick Reference: What to Train

**Minimum Training (for all hypotheses):**
```bash
# 1. Baseline (required for all comparisons)
python train.py --episodes 600 --device cuda --name baseline

# 2. H1 (required for H4)
python experiments/h1_option_structure/train.py --episodes 600 --device cuda

# 3-6. Independent variants (can run in parallel)
python experiments/h2_learned_terminations/train.py --episodes 600 --device cuda
python experiments/h3_prompt_headers/train.py --episodes 600 --device cuda
python experiments/h5_state_ablation/train.py --episodes 600 --device cuda
python experiments/h6_transition_reward/train.py --episodes 600 --device cuda
python experiments/h7_hybrid_bert/train.py --episodes 600 --device cuda
```

**Total Models to Train: 6** (H2 can use baseline only)
- Baseline (hierarchical, standard BERT)
- H1 (flat)
- H3 (minimal prompts)
- H5 (state ablation)
- H6 (no transition rewards)
- H7 (hybrid BERT)

**H2 requires NO new training** - it evaluates baseline model's learned termination behavior

**H4 requires NO new training** - it compares Baseline vs H1

## Evaluation Dependencies

After training, evaluations are independent:

```bash
# Each evaluation only needs its own model + baseline
python experiments/h1_option_structure/evaluate.py --experiment-dir <H1_DIR>
python experiments/h2_learned_terminations/evaluate.py --experiment-dir <H2_DIR>
python experiments/h3_prompt_headers/evaluate.py --experiment-dir <H3_DIR>
python experiments/h4_training_stability/evaluate.py --hierarchical-dir <BASELINE_DIR> --flat-dir <H1_DIR>
python experiments/h5_state_ablation/evaluate.py --experiment-dir <H5_DIR>
python experiments/h6_transition_reward/evaluate.py --experiment-dir <H6_DIR>
python experiments/h7_hybrid_bert/evaluate.py --experiment-dir <H7_DIR>
```

## Finding Model Directories

Use the helper script:
```bash
python experiments/find_experiments.py --save-paths
```

This creates `experiments/experiment_paths.json` with all model locations.

