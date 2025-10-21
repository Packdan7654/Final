# HRL Museum Dialogue Agent

Actor-Critic system for museum dialogue using hierarchical RL.

## Overview

Implements the approach from `paper.tex`:
- **Options**: Explain, Ask, Transition, Conclude
- **Actor-Critic**: Learns option policies and termination
- **DialogueBERT**: Intent recognition
- **Mistral LLM**: Local utterance generation

## Setup

### 1. Install Mistral
```bash
# Install Ollama from https://ollama.ai
ollama pull mistral
ollama serve
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

## Training

```bash
# Quick test (10 episodes, ~2 min)
python train.py --episodes 10

# Full training (200 episodes, ~2 hours)
python train.py --episodes 200

# GPU training
python train.py --episodes 500 --device cuda
```

Trained model saved to `models/trained_agent.pt`

## Architecture (from paper.tex)

**State Space (149-d)**:
- Focus: 9-d (8 exhibits + no-focus)
- History: 12-d (8 exhibits + 4 options)
- Intent: 64-d (DialogueBERT projection)
- Context: 64-d (DialogueBERT projection)

**Options**: Explain, Ask, Transition, Conclude

**Reward**: Engagement (dwell) + Novelty (facts) + Deliberation cost

## Files

```
train.py              - Train agent
src/
  agent/              - Actor-Critic implementation
  environment/        - SMDP environment
  training/           - Training loop
  simulator/          - User simulation
  utils/              - DialogueBERT, LLM handler, prompts
paper.tex             - Research questions and approach
```
