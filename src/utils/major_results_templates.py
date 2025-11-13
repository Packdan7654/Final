"""
README Templates for Major Results

Provides README templates for each model variation in major_results/.
Each template includes model description, key differences, configuration, and metrics.
"""

from pathlib import Path


def get_readme_template(model_name: str, metadata: dict = None) -> str:
    """
    Get README template for a model variation.
    
    Args:
        model_name: Model name (normalized)
        metadata: Optional metadata dictionary from training
        
    Returns:
        README content as string
    """
    templates = {
        'baseline': get_baseline_readme(metadata),
        'h1_flat_policy': get_h1_readme(metadata),
        'h2_learned_terminations': get_h2_readme(metadata),
        'h3_minimal_prompts': get_h3_readme(metadata),
        'h5_state_ablation': get_h5_readme(metadata),
        'h6_transition_reward': get_h6_readme(metadata),
        'h7_hybrid_bert': get_h7_readme(metadata),
    }
    
    return templates.get(model_name, get_default_readme(model_name, metadata))


def get_baseline_readme(metadata: dict = None) -> str:
    """Baseline model README template."""
    return f"""# Baseline: Hierarchical Option-Critic with Standard BERT

## Model Description

This is the baseline hierarchical reinforcement learning (HRL) system using the Option-Critic architecture (Bacon et al., 2017) built on Actor-Critic with TD(0) learning.

### Architecture
- **Algorithm**: Option-Critic (hierarchical Actor-Critic)
- **State Representation**: 149-dimensional vector
  - `f_t`: Focus vector (9-d for 8 exhibits)
  - `h_t`: Dialogue history (12-d: exhibit completion + option usage)
  - `i_t`: Intent embedding (64-d, projected from 768-d standard BERT)
  - `c_t`: Dialogue context (64-d, projected from 768-d standard BERT)
- **BERT Mode**: Standard BERT (no turn/role embeddings) for both `i_t` and `c_t`
- **Options**: Explain, AskQuestion, OfferTransition, Conclude
- **Subactions**: 3 per option (12 total primitive actions)
- **Termination**: Learned termination functions (Option-Critic style)

### Key Features
- Hierarchical policy structure (option-level + intra-option policies)
- Learned termination functions for adaptive option duration
- Standard BERT embeddings for intent and context
- Action masking (prevents Conclude until minimum facts/exhibits covered)

### Training Configuration
{_format_metadata(metadata)}

## Results Location

- **Training Results**: `training/`
- **Evaluation Metrics**: `evaluation/`
- **Visualizations**: `visualizations/`
- **Consolidated Metrics**: `metrics/`

## Key Metrics

See `metrics/training_metrics.json` for comprehensive metrics including:
- Episode returns and coverage
- Option usage and durations
- Policy and value losses
- Convergence analysis

## Comparison

This baseline is compared against:
- **H1**: Flat Actor-Critic (no hierarchical structure)
- **H2**: Fixed-duration Explain option (no learned terminations)
- **H3**: Minimal prompts (no structured headers)
- **H5**: State ablation (dialogue-act-only state)
- **H6**: No transition rewards
- **H7**: Hybrid BERT (DialogueBERT for context, standard BERT for intent)
"""


def get_h1_readme(metadata: dict = None) -> str:
    """H1: Flat Policy README template."""
    return f"""# H1: Flat Actor-Critic (No Hierarchical Structure)

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
{_format_metadata(metadata)}

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
"""


def get_h2_readme(metadata: dict = None) -> str:
    """H2: Learned Terminations README template."""
    return f"""# H2: Learned Terminations Analysis

## Model Description

This variant analyzes the learned termination behavior of the baseline model. **No new training is required** - this uses the baseline model's learned termination functions.

### Architecture
- **Same as Baseline**: Uses baseline model's learned termination functions
- **Analysis Focus**: Correlation between Explain option duration and dwell time, termination timing after intent changes

### Key Differences from Baseline
- **No differences in architecture**: Uses baseline model
- **Analysis only**: Evaluates baseline's learned termination behavior
- **Optional variant**: Fixed-duration Explain option (3 turns) can be trained for comparison

### Hypothesis (H2)
Learned termination functions adapt Explain option duration based on visitor engagement (dwell time) and intent changes. We expect:
- Positive correlation between Explain segment length and mean dwell within segment
- Earlier termination after detected intent changes away from "explain"

### Training Configuration
{_format_metadata(metadata)}

**Note**: H2 does not require new training. It evaluates the baseline model's learned termination behavior.

## Results Location

- **Evaluation Metrics**: `evaluation/` (from baseline model analysis)
- **Visualizations**: `visualizations/`
- **Consolidated Metrics**: `metrics/`

## Key Metrics

See `evaluation/h2_metrics.json` for:
- Correlation between Explain duration and dwell time
- Time-to-termination after intent changes
- Termination rate statistics
- Dwell correlation confidence intervals
"""


def get_h3_readme(metadata: dict = None) -> str:
    """H3: Minimal Prompts README template."""
    return f"""# H3: Minimal Prompts (No Structured Headers)

## Model Description

This variant tests whether structured prompt headers improve realization quality. It uses minimal prompts without structured formatting, rules, and examples, while keeping essential information.

### Architecture
- **Same as Baseline**: Hierarchical Option-Critic
- **State Representation**: Same as baseline (149-d)
- **BERT Mode**: Standard BERT (same as baseline)
- **Key Difference**: Minimal prompts instead of structured headers

### Key Differences from Baseline
- **Prompt Format**: Minimal prompts without:
  - Structured headers (CURRENT EXHIBIT:, VISITOR SAID:, etc.)
  - Verbose rules and examples
  - Progress percentages
  - Detailed fact ID warnings
- **Essential Information Retained**:
  - Available facts (top 5 for Explain actions)
  - Facts already used (to avoid repetition)
  - Dialogue history (last 4 utterances)
  - Target exhibit (for transitions)
  - Coverage stats (for transitions)

### Hypothesis (H3)
Slot-filled prompt headers tied to the active option will improve the next turn's realization: higher faithfulness to the intended move, tighter grounding to the exhibit KB, and less repetition.

### Training Configuration
{_format_metadata(metadata)}

## Results Location

- **Training Results**: `training/`
- **Evaluation Metrics**: `evaluation/`
- **Visualizations**: `visualizations/`
- **Consolidated Metrics**: `metrics/`

## Key Metrics

See `evaluation/h3_metrics.json` for:
- Novel fact coverage |F_used|/|F_total|
- Repetition ratio
- Grounding precision/recall
- Hallucination rate
- On-plan compliance
- Act classifier agreement
"""


def get_h5_readme(metadata: dict = None) -> str:
    """H5: State Ablation README template."""
    return f"""# H5: State Ablation (Dialogue-Act-Only State)

## Model Description

This variant tests whether a compact dialogue-act-only state representation can maintain performance compared to full DialogueBERT embeddings.

### Architecture
- **Same as Baseline**: Hierarchical Option-Critic
- **State Representation**: Ablated to ~20-d (vs 149-d baseline)
  - `f_t`: Focus vector (9-d)
  - `h_t`: Dialogue history (12-d)
  - `a_t`: Dialogue act one-hot (6-d) - **replaces DialogueBERT embeddings**
- **BERT Mode**: Not used (replaced by dialogue act classifier)

### Key Differences from Baseline
- **State Dimension**: ~20-d instead of 149-d
- **No DialogueBERT**: Replaced with dialogue act classification
- **Dialogue Act Classifier**: Rule-based classifier for intent categories
- **Compression Ratio**: ~86% reduction in state dimension

### Hypothesis (H5)
Replacing DialogueBERT's full 149-d state with a low-dimensional dialogue-act trace will reduce performance. If lexical semantics are essential, the ablated state should show lower return, poorer coverage, and weaker responsiveness.

### Training Configuration
{_format_metadata(metadata)}

## Results Location

- **Training Results**: `training/`
- **Evaluation Metrics**: `evaluation/`
- **Visualizations**: `visualizations/`
- **Consolidated Metrics**: `metrics/`

## Key Metrics

See `evaluation/h5_metrics.json` for:
- State dimension and compression ratio
- Episode returns (compared to baseline)
- Coverage |F_used|/|F_total|
- Act consistency
- Responsiveness rate
- Mean/median dwell time
"""


def get_h6_readme(metadata: dict = None) -> str:
    """H6: Transition Reward Ablation README template."""
    return f"""# H6: No Transition Rewards

## Model Description

This variant tests the importance of transition-acceptance shaping rewards by removing the transition sufficiency bonus and insufficiency penalty.

### Architecture
- **Same as Baseline**: Hierarchical Option-Critic
- **State Representation**: Same as baseline (149-d)
- **BERT Mode**: Standard BERT (same as baseline)
- **Key Difference**: Transition rewards ablated (set to 0)

### Key Differences from Baseline
- **No Transition Sufficiency Bonus**: Removed reward for successful transitions after sufficient facts
- **No Transition Insufficiency Penalty**: Removed penalty for transitions with too few facts
- **Other Rewards Unchanged**: Engagement, novelty, responsiveness, conclude rewards remain

### Hypothesis (H6)
Removing the transition-acceptance shaping term from the reward will degrade pacing. Without this signal, we expect earlier, less-informed transitions, lower dwell prior to transitions, and reduced exhibit coverage.

### Training Configuration
{_format_metadata(metadata)}

## Results Location

- **Training Results**: `training/`
- **Evaluation Metrics**: `evaluation/`
- **Visualizations**: `visualizations/`
- **Consolidated Metrics**: `metrics/`

## Key Metrics

See `evaluation/h6_metrics.json` for:
- Transition success rate
- Mean dwell before transition
- Confusion responses after transitions
- Premature transition rate
- Transition timing distribution
- Facts shared per exhibit before leaving
"""


def get_h7_readme(metadata: dict = None) -> str:
    """H7: Hybrid BERT README template."""
    return f"""# H7: Hybrid BERT (Standard for Intent, DialogueBERT for Context)

## Model Description

This variant tests whether DialogueBERT's turn-aware and role-aware embeddings improve multi-turn dialogue context understanding when used for `c_t`, while keeping standard BERT for single-utterance intent (`i_t`).

### Architecture
- **Same as Baseline**: Hierarchical Option-Critic
- **State Representation**: Same as baseline (149-d)
- **BERT Mode**: Hybrid approach
  - `i_t` (intent): Standard BERT (no turn/role embeddings) - **same as baseline**
  - `c_t` (context): DialogueBERT (with turn/role embeddings) - **different from baseline**

### Key Differences from Baseline
- **Intent Embedding (`i_t`)**: Standard BERT (same as baseline)
- **Context Embedding (`c_t`)**: DialogueBERT with:
  - Turn embeddings: Track turn position in dialogue (0-indexed)
  - Role embeddings: Distinguish user (0) vs system/agent (1)
- **Rationale**: Multi-turn context benefits from turn/role awareness, while single-utterance intent may not

### Hypothesis (H7)
Using DialogueBERT's turn-aware and role-aware embeddings for multi-turn dialogue context (`c_t`) while keeping standard BERT for single-utterance intent (`i_t`) will improve dialogue coherence and context-dependent responses.

### Training Configuration
{_format_metadata(metadata)}

## Results Location

- **Training Results**: `training/`
- **Evaluation Metrics**: `evaluation/`
- **Visualizations**: `visualizations/`
- **Consolidated Metrics**: `metrics/`

## Key Metrics

See `evaluation/h7_metrics.json` for:
- Dialogue coherence (reference resolution accuracy)
- Context-dependent question answering rate
- Response appropriateness (contextual vs generic)
- Contradiction rate
- Embedding similarity between consecutive turns
- Reference tracking accuracy
"""


def get_default_readme(model_name: str, metadata: dict = None) -> str:
    """Default README template for unknown models."""
    return f"""# {model_name}

## Model Description

Model variation: {model_name}

### Training Configuration
{_format_metadata(metadata)}

## Results Location

- **Training Results**: `training/`
- **Evaluation Metrics**: `evaluation/`
- **Visualizations**: `visualizations/`
- **Consolidated Metrics**: `metrics/`
"""


def _format_metadata(metadata: dict = None) -> str:
    """Format metadata dictionary for README."""
    if not metadata:
        return "Not available"
    
    lines = []
    if 'episodes' in metadata:
        lines.append(f"- **Episodes**: {metadata['episodes']}")
    if 'learning_rate' in metadata:
        lines.append(f"- **Learning Rate**: {metadata['learning_rate']}")
    if 'gamma' in metadata:
        lines.append(f"- **Gamma**: {metadata['gamma']}")
    if 'device' in metadata:
        lines.append(f"- **Device**: {metadata['device']}")
    if 'bert_mode' in metadata:
        lines.append(f"- **BERT Mode**: {metadata['bert_mode']}")
    if 'timestamp' in metadata:
        lines.append(f"- **Training Date**: {metadata['timestamp']}")
    
    return "\n".join(lines) if lines else "Not available"


def create_readme_file(model_dir: Path, model_name: str, metadata: dict = None):
    """
    Create README.md file for a model.
    
    Args:
        model_dir: Path to model directory
        model_name: Model name (normalized)
        metadata: Optional metadata dictionary
    """
    readme_content = get_readme_template(model_name, metadata)
    readme_path = model_dir / "README.md"
    
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    return readme_path

