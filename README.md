# HRL Museum Dialogue Agent

Hierarchical Reinforcement Learning for adaptive museum dialogue.

## Quick Start

1. **Set API Key**
   ```bash
   $env:GROQ_API_KEY='your_key_here'  # Windows PowerShell
   export GROQ_API_KEY='your_key_here'  # Linux/Mac
   ```

2. **Train**
   ```bash
   python train.py --episodes 600 --device cuda
   ```

3. **Results**
   - Training logs: `training_logs/experiments/exp_XXX/`
   - Evaluation: Automatically generated after training
   - Parameterization analysis: Run `python analyze_parameterization.py <exp_num>`

## Documentation

- **[TRAINING.md](TRAINING.md)** - How to train and configure reward weights
- **[EVALUATION.md](EVALUATION.md)** - Understanding evaluation results

## Project Structure

```
Thesis_HRL/
├── train.py                    # Unified training script
├── analyze_parameterization.py  # Reward weight analysis
├── create_evaluation_plots.py  # Evaluation plot generator
│
├── src/                        # Core code
│   ├── environment/            # Museum environment
│   ├── agents/                 # Actor-Critic agent
│   ├── training/               # Training loop
│   ├── simulator/              # Visitor simulator
│   └── utils/                  # LLM, DialogueBERT, logging
│
└── training_logs/experiments/  # All results
```

## Technical Details

- **RL Algorithm**: Actor-Critic with TD(0)
- **Options**: Explain, Ask, Transition, Conclude (4 high-level strategies)
- **State**: 149-dim (gaze features + dialogue history + intent)
- **Reward**: Weighted sum (engagement + novelty + responsiveness + conclude)
- **LLM**: Groq API (Llama 3.1 8B)

## Reward Configuration

Reward weights are configurable via command line:

```bash
# Paper baseline
python train.py --w-engagement 1.0 --w-novelty 0.5

# Encourage more facts
python train.py --w-novelty 1.0 --novelty-per-fact 0.2
```

See [TRAINING.md](TRAINING.md) for full details.

---

**For thesis details, see `paper.tex`**
