# Baseline Experiment (Reference)

The baseline experiment uses the standard HRL system as implemented in the main codebase.

## Training

Use the main training script:

```bash
python train.py --episodes 600 --name baseline --device cuda
```

This trains the full system with:
- Full DialogueBERT state (149-d)
- All reward components including transition shaping
- Standard hierarchical actor-critic architecture

## Evaluation

After training, evaluate using the standard evaluation tools:

```bash
python tools/create_evaluation_plots.py --experiment <baseline_experiment_dir>
```

## Comparison

The baseline serves as the reference point for comparing:
- **H5**: State ablation (dialogue-act vs DialogueBERT)
- **H6**: Transition reward shaping (with vs without)

Use `experiments/shared/comparison_tools.py` to generate comparison reports.

