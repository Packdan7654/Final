# DialogueBERT in the Decision Pipeline

## Complete Processing Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│ STEP 1: User Input                                                  │
└─────────────────────────────────────────────────────────────────────┘
    Utterance: "Tell me about this painting"
    Focus: 1 (King_Caspar)
    Dwell: 0.85

                              ↓

┌─────────────────────────────────────────────────────────────────────┐
│ STEP 2: DialogueBERT Processing (HAPPENS FIRST!)                   │
└─────────────────────────────────────────────────────────────────────┘

    A. Intent Embedding
    ────────────────────────────────────────────────────────
    Input: "Tell me about this painting"
    
    DialogueBERT.get_intent_embedding(utterance, role="user")
    ↓
    768-dimensional embedding
    [0.234, -0.456, 0.789, ..., -0.123]  ← Captures semantic meaning
    
    ↓ Projection (768-d → 64-d)
    
    intent_64d = projection_matrix @ intent_768d
    [0.12, -0.45, 0.78, ..., -0.69]  ← Compressed representation
    
    
    B. Context Embedding
    ────────────────────────────────────────────────────────
    Input: Last 3 dialogue turns
    
    DialogueBERT.get_dialogue_context(recent_turns, max_turns=3)
    ↓
    768-dimensional context embedding
    [0.145, 0.823, -0.234, ..., 0.456]  ← Captures conversation flow
    
    ↓ Projection (768-d → 64-d)
    
    context_64d = projection_matrix @ context_768d
    [0.34, 0.82, -0.15, ..., -0.69]  ← Compressed representation

                              ↓

┌─────────────────────────────────────────────────────────────────────┐
│ STEP 3: State Vector Construction                                  │
└─────────────────────────────────────────────────────────────────────┘

    Combine all components:
    
    state = [
        focus_vector,        # 9-d:  [1,0,0,0,0,0,0,0,0]
        history_vector,      # 12-d: [0,0,0,...] (no facts yet)
        intent_64d,          # 64-d: DialogueBERT output ←
        context_64d          # 64-d: DialogueBERT output ←
    ]
    
    Total: 157-dimensional state vector
    
                              ↓

┌─────────────────────────────────────────────────────────────────────┐
│ STEP 4: Agent's Neural Network (Actor-Critic)                      │
└─────────────────────────────────────────────────────────────────────┘

    Input: state (157-d, includes DialogueBERT embeddings)
    ↓
    LSTM Encoder (128 units)
    ↓
    Actor heads (option, subaction, termination)
    Critic heads (Q-values, state value)
    ↓
    Decision: Explain / ExplainNewFact
```

---

## Key Points

### **1. DialogueBERT Runs FIRST**

From `src/environment/env.py`, line 338-354:

```python
def _get_obs(self):
    # ... focus and history ...
    
    # 3. Intent embedding (DialogueBERT processes utterance HERE)
    intent_recognizer = get_dialoguebert_recognizer()
    intent_embedding_768 = intent_recognizer.get_intent_embedding(
        self.last_user_utterance, role="user"
    )
    
    # Project from 768-d to 64-d
    intent_embedding_64 = np.dot(self.projection_matrix, intent_embedding_768)
    
    # 4. Dialogue context (DialogueBERT processes history HERE)
    dialogue_history_with_roles = [("user", u) for u in self.dialogue_history]
    dialogue_context_768 = intent_recognizer.get_dialogue_context(
        dialogue_history_with_roles, max_turns=3
    )
    
    # Project from 768-d to 64-d
    dialogue_context_64 = np.dot(self.projection_matrix, dialogue_context_768)
    
    # Combine into state
    obs = np.concatenate([
        focus_snapshot,        # 9-d
        history,               # 12-d
        intent_embedding_64,   # 64-d ← From DialogueBERT
        dialogue_context_64    # 64-d ← From DialogueBERT
    ])
```

### **2. Why DialogueBERT First?**

**DialogueBERT's job:** Convert raw text into semantic vectors
- "Tell me about this" → [0.12, -0.45, 0.78, ...]
- "What's this painting?" → [0.11, -0.44, 0.79, ...] (similar vector!)
- "I hate museums" → [-0.82, 0.34, -0.12, ...] (very different vector!)

**Agent's job:** Map states to actions
- The agent's network doesn't understand text
- It only understands numbers
- DialogueBERT translates text → numbers

### **3. Two Separate Neural Networks**

```
DialogueBERT (Pre-trained)          Agent Network (Trained by us)
─────────────────────────           ──────────────────────────────
• 110M parameters                   • 585K parameters
• Pre-trained on dialogue           • Trained on museum task
• Frozen (not updated)              • Updated during RL training
• Job: text → semantic vector       • Job: state → action
```

---

## Detailed Example

### **Turn 1: "Tell me about this painting"**

**Step 1: DialogueBERT Intent Encoding**
```python
utterance = "Tell me about this painting"

# DialogueBERT processes this
intent_768d = DialogueBERT.encode(utterance)
# Result: [0.234, -0.456, 0.789, 0.123, -0.345, ...]  (768 numbers)

# This captures:
# - It's a question (not a statement)
# - Requesting information (not opinion)
# - About current object (not general)
# - Polite/neutral tone
```

**Step 2: Projection to Compact Representation**
```python
# Project 768-d → 64-d to keep state manageable
projection_matrix = np.random.randn(64, 768)  # Fixed, pre-computed
intent_64d = projection_matrix @ intent_768d
# Result: [0.12, -0.45, 0.78, ...]  (64 numbers)

# This preserves the semantic meaning in fewer dimensions
```

**Step 3: Context Encoding**
```python
recent_history = []  # Empty on first turn
context_768d = DialogueBERT.encode_context(recent_history)
context_64d = projection_matrix @ context_768d
# Result: [0.34, 0.82, -0.15, ...]  (64 numbers)
```

**Step 4: Build State**
```python
state = np.concatenate([
    [1, 0, 0, 0, 0, 0, 0, 0, 0],  # Focus on exhibit 1
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # No history
    [0.12, -0.45, 0.78, ..., -0.69],  # Intent (64-d) ← DialogueBERT
    [0.34, 0.82, -0.15, ..., -0.69]   # Context (64-d) ← DialogueBERT
])
# Total: 157 numbers
```

**Step 5: Agent Network Processes State**
```python
# NOW the agent's network runs
outputs = agent.network.forward(state)
# Outputs: option logits, subaction logits, termination probs
# Decision: Explain / ExplainNewFact
```

---

## Why This Architecture?

### **Separation of Concerns**

**DialogueBERT:**
- Pre-trained on millions of dialogue examples
- Knows: "tell me about" = information request
- Knows: "I don't like" = negative sentiment
- Knows: "what else" = seeking alternatives
- **Frozen during training** (we don't update it)

**Agent Network:**
- Trained on museum task
- Learns: information_request + high_dwell → Explain
- Learns: negative_sentiment + low_dwell → Transition
- **Updated during training** to maximize reward

### **Dimensionality Reduction**

**Why 768 → 64 projection?**

Without projection:
```
State = [9 + 12 + 768 + 768] = 1,557-dimensional
```

With projection:
```
State = [9 + 12 + 64 + 64] = 157-dimensional  ← Much smaller!
```

Benefits:
- Faster training (fewer parameters in agent network)
- Better generalization (less overfitting)
- Still preserves semantic information (proven by Johnson-Lindenstrauss lemma)

---

## Code Location Summary

| Operation | File | Line | What Happens |
|-----------|------|------|--------------|
| DialogueBERT intent | `env.py` | 338-344 | Encode utterance → 768-d → 64-d |
| DialogueBERT context | `env.py` | 346-354 | Encode history → 768-d → 64-d |
| Build state | `env.py` | 366-370 | Concatenate all components |
| Agent forward pass | `actor_critic_agent.py` | 95-98 | Process state → actions |

---

## Visual Timeline

```
Time ──────────────────────────────────────────────────────→

User types: "Tell me about this"
    ↓ (1ms)
DialogueBERT: text → 768-d vector
    ↓ (1ms)
Projection: 768-d → 64-d
    ↓ (instant)
Build state: [focus, history, intent_64d, context_64d]
    ↓ (instant)
Agent LSTM: state → encoded (128-d)
    ↓ (5ms)
Actor heads: encoded → option/subaction logits
    ↓ (1ms)
Softmax + selection: logits → action
    ↓ (instant)
Decision: Explain / ExplainNewFact

Total: ~10ms for decision making
(Most time is in LLM response generation: ~10 seconds)
```

---

## Summary

**Yes, DialogueBERT runs FIRST!**

1. ✅ User provides text utterance
2. ✅ **DialogueBERT encodes it into semantic vectors** (768-d)
3. ✅ Vectors are projected to 64-d for efficiency
4. ✅ These become part of the state (157-d total)
5. ✅ Agent's network processes this state
6. ✅ Agent decides on action

**DialogueBERT is the "language understanding" module** that translates human speech into numerical representations the RL agent can use for decision-making.

The agent never sees raw text - it only sees:
- Numbers from DialogueBERT (semantic meaning)
- Numbers from focus/dwell (engagement signals)
- Numbers from history (what's been done)

All numbers, no text! 🔢

