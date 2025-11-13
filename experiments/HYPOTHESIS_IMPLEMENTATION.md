# Hypothesis Implementation Guide

This document maps each hypothesis from `paper.tex` to its experimental implementation, explaining the theoretical foundation, code modifications, and key metrics.

---

## H1: Option Structure vs Flat Policy

### Hypothesis (paper.tex, lines 882-884)
> An option-based manager will outperform a flat policy on long-horizon objectives. Concretely, we expect higher episodic return with lower variance, longer coherent stretches under a chosen strategy, and fewer needless switches between strategies when both are trained under the same rewards and prompts.

### Theoretical Foundation
**Hierarchical RL Theory (Sutton et al. 1999, Bacon et al. 2017):**
- Options provide temporal abstraction, allowing the agent to commit to a strategy over multiple timesteps
- This reduces the effective horizon for credit assignment and enables more stable learning
- Flat policies must learn primitive actions directly, leading to higher variance and less coherent behavior

**Expected Outcome:**
- Hierarchical system should show longer coherent spans (consecutive turns under same option)
- Lower switch rates (fewer unnecessary strategy changes)
- Higher mean return with lower variance

### Baseline Architecture Justification
**See [BASELINE_JUSTIFICATION.md](BASELINE_JUSTIFICATION.md) for detailed justification.**

**Summary:**
- **Hierarchical**: Option-Critic (Actor-Critic + options)
- **Flat**: Standard Actor-Critic (same base algorithm, no options)
- **Relationship**: Option-Critic = Actor-Critic + option structure
- Standard baseline used in major HRL papers (Bacon et al. 2017, Vezhnevets et al. 2017)
- Fair comparison: identical base algorithm, encoder, hyperparameters, and training procedure
- Only difference: policy structure (hierarchical options vs flat), isolating the structure effect

### Implementation

**Code Location:** `experiments/h1_option_structure/train.py`

**What It Does:**
1. Uses `FlatTrainingLoop` which wraps `FlatDialogueEnv` (from `src/flat_rl/env.py`)
2. `FlatDialogueEnv` flattens the hierarchical action space into a single discrete space:
   - Maps all (option, subaction) pairs to flat indices: `[Explain/ExplainNewFact, Explain/RepeatFact, ..., Conclude/WrapUp]`
   - Total: 12 flat actions (4 options × 3 subactions each)
3. Agent uses `FlatActorCriticAgent` with a single policy head over all flat actions
4. No option-level abstraction or learned terminations

**Key Code Changes:**
```python
# src/flat_rl/env.py
class FlatDialogueEnv(MuseumDialogueEnv):
    def __init__(self, ...):
        # Build flat action map: [(option_idx, option_name, subaction_idx, subaction_name), ...]
        self.flat_action_map = self._build_flat_action_map()
        self.action_space = spaces.Discrete(len(self.flat_action_map))  # Single discrete space
    
    def step(self, action_index: int):
        # Map flat index back to (option, subaction) for compatibility
        option_idx, option_name, subaction_idx, subaction_name = self.flat_action_map[action_index]
        return super().step({"option": option_idx, "subaction": subaction_idx, "terminate_option": False})
```

**Baseline Comparison:**
- Baseline uses hierarchical `MuseumDialogueEnv` with option-level policy and learned terminations
- Both use same simulator, rewards, and prompts (controlled comparison)

### Key Metrics (paper.tex, lines 912-913)

**Primary Metrics:**
- **Mean episodic return** (with variance) - `common_metrics.mean_return`, `std_return`
- **Average coherent-span length** - `h1_metrics.mean_coherent_span` (consecutive turns under same option)
- **Switch rate** - `h1_metrics.switch_rate_per_100_turns` (switches per 100 turns)

**Statistical Analysis:**
- Paired t-test or Wilcoxon signed-rank test (per-seed pairing)
- Effect size calculation

**Evaluation Code:** `experiments/h1_option_structure/evaluate.py`
- Computes option coherence by tracking consecutive turns under same option
- Calculates switch rate from option transitions
- Extracts from turn-by-turn logs in `logs/monitor_*.json`

**Thesis Writing Note:**
When reporting H1 results, cite the baseline justification:
- "We compare the hierarchical option-based system against a flat TD(0) Actor-Critic baseline, following standard practice in hierarchical RL (Bacon et al. 2017, Vezhnevets et al. 2017). The flat baseline uses the same encoder architecture, training algorithm, and hyperparameters as the hierarchical system, ensuring a fair comparison that isolates the effect of hierarchical structure."

---

## H2: Learned Terminations for Explain

### Hypothesis (paper.tex, lines 886-888)
> Learned terminations for Explain will track engagement and intent: when dwell is high and the visitor's intent supports explanation, Explain should persist; when dwell falls or intent shifts, it should end sooner. We expect a positive correlation between dwell and Explain duration, and earlier terminations following detected intent changes.

### Theoretical Foundation
**Option-Critic Architecture (Bacon et al. 2017):**
- Termination functions learn when to end an option based on state
- Should adapt to engagement signals (dwell time) and visitor intent
- Fixed-duration options cannot adapt to context, leading to suboptimal timing

**Expected Outcome:**
- Positive correlation (ρ > 0) between Explain segment length and mean dwell within segment
- Earlier terminations after intent changes (compared to fixed-duration baseline)

### Implementation

**Code Location:** `experiments/h2_learned_terminations/env.py`, `train.py`

**What It Does:**
1. `H2FixedDurationEnv` extends `MuseumDialogueEnv` to enforce fixed-duration Explain
2. Tracks `explain_turn_count` and forces termination after `fixed_explain_duration` turns (default: 3)
3. Baseline uses learned termination functions from `ActorCriticAgent` that adapt to state

**Key Code Changes:**
```python
# experiments/h2_learned_terminations/env.py
class H2FixedDurationEnv(MuseumDialogueEnv):
    def step(self, action_dict):
        if option == "Explain":
            self.explain_turn_count += 1
            if self.explain_turn_count >= self.fixed_explain_duration:
                action_dict["terminate_option"] = True  # Force termination
        else:
            self.explain_turn_count = 0  # Reset on option switch
```

**Baseline Comparison:**
- Baseline: `ActorCriticAgent` learns termination probabilities via `termination_loss` (encourages termination when advantage < 0)
- H2 variant: Fixed 3-turn duration regardless of dwell or intent

### Key Metrics (paper.tex, lines 915-916)

**Primary Metrics:**
- **Correlation (ρ)** between Explain segment length (τ) and mean dwell within segment - `h2_metrics.dwell_correlation`
- **Time-to-termination** after intent change vs matched no-shift periods
- **Explain durations** - `h2_metrics.explain_durations` (list of segment lengths)
- **Dwell during Explain** - `h2_metrics.dwell_during_explain` (mean dwell per segment)

**Visualization:**
- Plot: duration vs dwell with confidence bands
- Report correlations with confidence intervals

**Evaluation Code:** `experiments/h2_learned_terminations/evaluate.py`
- Tracks Explain segments from turn data
- Computes correlation using `np.corrcoef()`
- Aligns segments with dwell time from simulator

---

## H3: Prompt Headers and Local Realization

### Hypothesis (paper.tex, lines 891-893)
> Slot-filled prompt headers tied to the active option will improve the next turn's realization: higher faithfulness to the intended move, tighter grounding to the exhibit KB, and less repetition. Concretely, we expect higher novel-fact coverage, a lower repetition ratio, a higher KB-citation/grounding rate with fewer off-KB claims, and better on-plan consistency with the chosen option/subaction.

### Theoretical Foundation
**Prompt Engineering & Constrained Generation:**
- Structured prompts with explicit constraints (fact IDs, exhibit focus, dialogue history) guide LLM generation
- Without headers, LLM relies on implicit context, leading to:
  - Hallucinations (claims not in KB)
  - Repetition (re-mentioning same facts)
  - Off-plan behavior (not following intended subaction)

**Expected Outcome:**
- Higher novel-fact coverage (more unique facts introduced)
- Lower repetition ratio (fewer re-mentions)
- Lower hallucination rate (fewer non-KB claims)
- Higher KB citation rate

### Implementation

**Code Location:** `experiments/h3_prompt_headers/train.py`

**What It Does:**
1. Monkey-patches `src.utils.dialogue_planner.build_prompt()` to use minimal prompts
2. **Minimal prompt** removes structured headers, verbose rules, and examples, but **keeps essential information**:
   - Current exhibit and visitor's message
   - Available facts (top 5 for Explain actions)
   - Facts already used (to avoid repetition)
   - Dialogue history (last 4 utterances for coherence)
   - Target exhibit (for transitions)
   - Coverage stats (for transitions)
3. **Baseline** uses full structured headers with:
   - Structured formatting (`CURRENT EXHIBIT:`, `VISITOR SAID:`, separators)
   - Dialogue history with emojis
   - Detailed fact tracking with warnings
   - Verbose rules and examples (✓/✗ examples)
   - Progress percentages
   - Explicit fact ID constraints

**Key Code Changes:**
```python
# experiments/h3_prompt_headers/train.py
def build_minimal_prompt(option, subaction, ex_id, last_utt, facts_all, facts_used, ...):
    """Minimal prompt without structured headers (H3 variant).
    
    Removes verbose formatting, rules, and examples, but keeps essential information:
    - Current exhibit and visitor's message
    - Available facts (for Explain actions)
    - Facts already used (to avoid repetition)
    - Dialogue history (for coherence)
    - Target exhibit (for transitions)
    """
    # Base context (plain text, no structured headers)
    prompt_parts = [
        f"You are a museum guide at: {ex_id.replace('_', ' ')}",
        f"Visitor said: \"{last_utt}\"",
        f"Action: {option} / {subaction}",
        ""
    ]
    
    # Add dialogue history (plain, no emojis)
    if dialogue_history:
        prompt_parts.append("Recent conversation:")
        for role, utterance in dialogue_history[-4:]:
            role_label = "Agent" if role == "agent" else "Visitor"
            prompt_parts.append(f"  {role_label}: \"{utterance}\"")
        prompt_parts.append("")
    
    # Add available facts (limited to 5, no verbose formatting)
    if option == "Explain" and subaction == "ExplainNewFact" and facts_all:
        prompt_parts.append("Available facts to share:")
        for fact in facts_all[:5]:
            prompt_parts.append(f"  {fact}")
        prompt_parts.append("")
        # Simple fact ID tracking (no warnings)
        fact_ids_mentioned = extract_fact_ids(dialogue_history)
        if fact_ids_mentioned:
            prompt_parts.append(f"Already mentioned fact IDs (don't repeat): {sorted(fact_ids_mentioned)}")
            prompt_parts.append("")
        prompt_parts.append("Share 1-3 facts that address what the visitor said. Include fact IDs in brackets like [TU_001].")
    
    # ... (similar for other options/subactions)
    
    return "\n".join(prompt_parts)

class H3TrainingLoop(HRLTrainingLoop):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Patch dialogue planner
        import src.utils.dialogue_planner as dp
        dp.build_prompt = build_minimal_prompt
```

**Baseline Comparison:**
- Baseline: Full structured prompts from `src/utils/dialogue_planner.py` with:
  - `_build_enhanced_context_section()` - Rich context with dialogue history
  - `build_explain_new_fact_prompt()` - Explicit fact ID constraints
  - Fact reuse warnings and KB grounding

### Key Metrics (paper.tex, lines 918-919)

**Primary Metrics (paper.tex, lines 918-919):**
- **Novel-fact coverage** - `h3_metrics.novel_fact_coverage` = |F_used| / |F_total|
- **Repetition ratio** - `h3_metrics.repetition_ratio` = (re-mentions beyond first) / (total fact mentions)
- **Grounding precision/recall** - Via KB aligner (true positives, false positives, false negatives)
- **Hallucination rate** - `h3_metrics.hallucination_rate` = (claims with no KB match) / (total claims)
- **On-plan compliance** - Agreement between intended subaction and realized utterance:
  - `ExplainNewFact` introduces a new fact ID
  - `Clarify/Repeat` avoid new IDs
  - `Ask*` ends with a question
- **Act classifier agreement** - Lightweight text-only classifier provides independent check that realized acts match the header
- **Header completeness correlation** - Share of slots filled (FOCUS, LAST_UTT, fact IDs) correlated with violations and hallucinations (negative correlation expected)

**Evaluation Code:** `experiments/h3_prompt_headers/evaluate.py`
- Extracts fact mentions from turn data (`facts_mentioned_in_utterance`)
- Tracks unique facts vs repeated facts
- Identifies hallucinations (facts not in KB) from `hallucinated_facts` in turn data

---

## H4: Training Stability with Actor-Critic

### Hypothesis (paper.tex, lines 895-897)
> With semi-Markov returns, actor-critic will train the hierarchy smoothly. Relative to a flat actor-critic (no options) and a fixed-duration hierarchy (one-turn options), we expect steadier learning curves (fewer spikes), smaller update magnitudes, and faster time-to-target reward, while preserving sensible option lengths.

### Theoretical Foundation
**Semi-Markov Decision Processes (SMDPs):**
- Options enable credit assignment over multi-step sequences
- Reduces variance in policy gradients compared to flat policies
- Hierarchical structure provides natural regularization

**Expected Outcome:**
- Smoother learning curves (lower variance of return differences)
- Smaller update magnitudes (L2-norm of parameter steps)
- Faster convergence (episodes to reach target return)

### Implementation

**Code Location:** `experiments/h4_training_stability/evaluate.py`

**What It Does:**
1. Compares training dynamics between:
   - **Hierarchical baseline** (from `train.py --name baseline`)
   - **Flat variant** (from H1 experiment)
2. Extracts learning curves from `logs/metrics_tracker_*.json` (episode returns)
3. Computes smoothness metrics (variance of return differences)
4. Calculates time-to-target (episodes to reach 80% of max return)

**Key Code:**
```python
# experiments/h4_training_stability/evaluate.py
def compute_h4_metrics(hierarchical_data, flat_data):
    # Learning curve smoothness (variance of differences)
    h_diffs = np.diff(h_returns)
    metrics['hierarchical']['curve_smoothness'] = np.std(h_diffs)
    
    # Time to target (episodes to reach 80% of max return)
    h_target = 0.8 * h_max
    h_time_to_target = next((i for i, r in enumerate(h_returns) if r >= h_target), len(h_returns))
```

**Data Sources:**
- Hierarchical: `training_logs/experiments/.../exp_XXX_baseline_*/logs/metrics_tracker_*.json`
- Flat: `training_logs/experiments/.../major_XXX_h1_flat_policy_*/logs/metrics_tracker_*.json`

### Key Metrics (paper.tex, lines 921-922)

**Primary Metrics:**
- **Learning curve smoothness** - `h4_metrics.hierarchical.curve_smoothness` (std of return differences)
- **Update magnitudes** - L2-norm of parameter steps, KL divergence per update (from trainer stats)
- **Time-to-target** - `h4_metrics.hierarchical.time_to_target` (episodes to reach fixed return threshold)
- **Option duration sanity** - Median τ per option (from baseline logs)

**Visualization:**
- Plot: Smoothed return vs updates (both hierarchical and flat)
- Compare variance and convergence speed

**Evaluation Code:** `experiments/h4_training_stability/evaluate.py`
- Loads both experiments and extracts learning curves
- Computes smoothness and convergence metrics
- Generates comparison report

---

## H5: Semantic State Ablation

### Hypothesis (paper.tex, lines 899-901)
> Replacing DialogueBERT's full 149-d state (Section 4.3) with a low-dimensional dialogue-act trace (one-hot over recent act types plus option usage) will reduce performance. If lexical semantics are essential, the ablated state should show lower return, poorer coverage, and weaker responsiveness relative to the full embedding setup.

### Theoretical Foundation
**State Representation in RL:**
- Rich semantic embeddings (DialogueBERT) capture lexical and contextual meaning
- Dialogue acts provide coarse-grained intent but lose semantic nuance
- Hypothesis tests if semantic richness is necessary for long-horizon dialogue

**Expected Outcome:**
- Lower mean return (reduced state information)
- Poorer coverage (less effective fact selection)
- Weaker responsiveness (cannot match visitor questions to appropriate responses)

### Implementation

**Code Location:** `experiments/h5_state_ablation/env.py`

**What It Does:**
1. `H5StateAblationEnv` replaces DialogueBERT embeddings with dialogue-act classification
2. State reduction: 149-d → ~20-d (for 5 exhibits)
   - Focus vector: (n_exhibits + 1) = 6-d
   - Dialogue history: (n_exhibits + 4) = 9-d
   - Dialogue act: 6-d one-hot (question, statement, acknowledgment, confusion, etc.)
3. Uses `DialogueActClassifier` to classify last user utterance

**Key Code Changes:**
```python
# experiments/h5_state_ablation/env.py
class H5StateAblationEnv(MuseumDialogueEnv):
    def _get_obs(self):
        # 1. Focus vector f_t (same as baseline)
        focus_snapshot = np.zeros(self.n_exhibits + 1)
        
        # 2. Dialogue history h_t (same as baseline)
        history = np.zeros(self.n_exhibits + len(self.options))
        
        # 3. Dialogue act a_t (H5: replaces DialogueBERT embeddings)
        act_type = self.act_classifier.classify(self.last_user_utterance, ...)
        act_vector = self.act_classifier.act_to_vector(act_type)  # 6-d one-hot
        
        # Concatenate: [f_t, h_t, a_t]
        obs = np.concatenate([focus_snapshot, history, act_vector])
```

**Baseline Comparison:**
- Baseline: Full 149-d state = [f_t(9-d), h_t(12-d), i_t(64-d), c_t(64-d)]
  - i_t: DialogueBERT intent embedding (64-d)
  - c_t: DialogueBERT dialogue context embedding (64-d)
- H5 variant: ~20-d state = [f_t(6-d), h_t(9-d), a_t(6-d)]

### Key Metrics (paper.tex, lines 924-925)

**Primary Metrics:**
- **Episodic return** - `common_metrics.mean_return` (vs baseline)
- **Coverage** - `common_metrics.mean_coverage` = |F_used| / |F_total|
- **Dwell averages** - Mean dwell time per episode
- **Act-consistency** - `h5_metrics.act_consistency` (agreement between predicted option/subaction and realized utterance)
- **Responsiveness rate** - `h5_metrics.responsiveness_rate` (fraction of visitor questions answered with new facts)

**Statistical Analysis:**
- Paired Wilcoxon test (per-seed pairing against full state)
- Large drops support H5 (semantic richness is essential)

**Evaluation Code:** `experiments/h5_state_ablation/evaluate.py`
- Uses `DialogueActClassifier` to verify act-consistency
- Computes state dimension reduction and compression ratio
- Extracts responsiveness from turn data (questions answered with new facts)

---

## H6: Transition Reward Shaping

### Hypothesis (paper.tex, lines 903-905)
> Removing the transition-acceptance shaping term from the reward (Section 4.7)—i.e., training without the probabilistic success bonus/penalty—will degrade pacing. Without this signal, we expect earlier, less-informed transitions, lower dwell prior to transitions, and reduced exhibit coverage compared to the shaped variant.

### Theoretical Foundation
**Reward Shaping in Hierarchical RL:**
- Transition rewards guide option-level decisions (when to switch exhibits)
- Insufficiency penalty discourages premature transitions (< 2 facts)
- Sufficiency reward encourages well-timed transitions (3+ facts)
- Without shaping, agent relies only on probabilistic acceptance from simulator

**Expected Outcome:**
- Lower transition success rate (more premature transitions)
- Lower mean dwell before transitions (less engagement)
- Fewer facts shared per exhibit before leaving
- Higher confusion responses after transitions

### Implementation

**Code Location:** `experiments/h6_transition_reward/env.py`

**What It Does:**
1. `H6TransitionRewardEnv` disables transition reward components:
   - `reward_transition_insufficiency` (penalty for < 2 facts)
   - `reward_transition_sufficiency` (reward for 3+ facts)
   - `reward_transition_frequency` (penalty for frequent transitions)
2. Keeps probabilistic transition acceptance from simulator (visitor can still reject)
3. Subtracts transition rewards from total reward in `step()`

**Key Code Changes:**
```python
# experiments/h6_transition_reward/env.py
class H6TransitionRewardEnv(MuseumDialogueEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_transition_rewards = False  # Disable shaping
    
    def step(self, action_dict):
        obs, reward, done, truncated, info = super().step(action_dict)
        
        # Remove transition reward components
        if not self.use_transition_rewards:
            transition_penalty = info.get("reward_transition_insufficiency", 0.0)
            transition_sufficiency = info.get("reward_transition_sufficiency", 0.0)
            transition_frequency = info.get("reward_transition_frequency", 0.0)
            reward -= (transition_penalty + transition_sufficiency + transition_frequency)
```

**Baseline Comparison:**
- Baseline: Full reward includes (from `src/environment/env.py`):
  ```python
  # Transition insufficiency penalty (per paper.tex Section 4.7)
  if facts_shared_at_current < 2:
      transition_insufficiency_penalty = self.w_transition_insufficiency * penalty_scale  # -0.20 * scale
  
  # Transition sufficiency reward
  if facts_shared_at_current >= 3:
      transition_sufficiency_reward = 0.15  # Positive reward
  ```
- H6 variant: These components are zeroed out

### Key Metrics (paper.tex, lines 927-928)

**Primary Metrics:**
- **Transition success rate** - `h6_metrics.transition_success_rate` (successful transitions / total attempts)
- **Mean dwell before transition** - Average dwell in turn preceding transition
- **Facts shared per exhibit** - `h6_metrics.avg_facts_before_transition` (before leaving)
- **Total coverage/return** - `common_metrics.mean_coverage`, `mean_return`
- **Confusion responses** - Rate of confusion after transitions (proxy for premature moves)

**Statistical Analysis:**
- Paired t-test (per-seed pairing)
- Quantifies effect of removing shaping term

**Evaluation Code:** `experiments/h6_transition_reward/evaluate.py`
- Tracks transition attempts and successes from turn data
- Extracts facts shared before transitions
- Computes reward component breakdown

---

## Summary: Experimental Design

### Controlled Variables
All experiments use:
- **Same simulator** (`Sim8Simulator`) with same persona seeds
- **Same knowledge graph** (`museum_knowledge_graph.json`)
- **Same exhibit itinerary** (fixed order)
- **Same reward structure** (except H6's transition reward toggle)
- **Same decoding settings** (LLM temperature, etc.)

### Independent Variables
- **H1**: Action space structure (hierarchical vs flat)
- **H2**: Termination mechanism (learned vs fixed)
- **H3**: Prompt structure (structured headers vs minimal)
- **H4**: Training dynamics (hierarchical vs flat - uses H1 data)
- **H5**: State representation (DialogueBERT vs dialogue acts)
- **H6**: Reward components (full vs ablated transition rewards)

### Dependent Variables (Metrics)
- Episodic return (mean, variance)
- Coverage ratio
- Option coherence (H1)
- Dwell correlation (H2)
- Hallucination rate (H3)
- Learning curve smoothness (H4)
- State dimension (H5)
- Transition success rate (H6)

### Statistical Analysis
- Paired tests (same seed, different model) for H1, H5, H6
- Correlation analysis for H2
- Comparison tests for H3, H4

---

## Running All Experiments

See `experiments/README.md` for step-by-step instructions on:
1. Training baseline and all variants
2. Running evaluation scripts
3. Generating comparison reports
4. Extracting metrics for thesis Results section

