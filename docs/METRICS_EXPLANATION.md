# HRL Museum Agent: Complete Metrics Reference Guide

This document explains every metric tracked in the HRL Museum Agent system, what it measures, how it's computed, and what "good" values look like.

---

## Table of Contents

1. [Core RL Metrics](#core-rl-metrics)
2. [Episode-Level Metrics](#episode-level-metrics)
3. [Turn-Level Metrics](#turn-level-metrics)
4. [Reward Component Metrics](#reward-component-metrics)
5. [Hierarchical Control Metrics](#hierarchical-control-metrics)
6. [Dialogue Quality Metrics](#dialogue-quality-metrics)
7. [Training Dynamics Metrics](#training-dynamics-metrics)
8. [Success Rate Metrics](#success-rate-metrics)
9. [Content Quality Metrics](#content-quality-metrics)
10. [Interpretation Guidelines](#interpretation-guidelines)

---

## Core RL Metrics

### Episode Return (Cumulative Reward)
**What it measures:** Total reward accumulated over an entire episode.

**How it's computed:** Sum of all per-turn rewards: $R = \sum_{t=1}^{T} r_t$

**Formula:** $R = r^{eng} + r^{nov} + r^{resp} + r^{trans} + r^{conclude}$ (summed over episode)

**What's good:**
- **Increasing trend:** Should improve over training episodes
- **Positive values:** Indicates agent is learning beneficial behaviors
- **Stability:** Lower variance indicates more consistent learning

**Interpretation:**
- High return = agent is successfully engaging users and providing valuable content
- Low/negative return = agent is making mistakes (deflecting, premature transitions, etc.)
- Increasing trend = agent is learning from experience

**Example:** Episode with return 9.684 means agent accumulated 9.684 total reward over the episode.

---

### Mean Return
**What it measures:** Average cumulative reward across all episodes.

**How it's computed:** $\bar{R} = \frac{1}{N} \sum_{i=1}^{N} R_i$ where $N$ is number of episodes

**What's good:**
- **Increasing over time:** Should improve as training progresses
- **Final 25% mean:** Should be higher than early episodes (shows learning)
- **Low standard deviation:** Indicates consistent performance

**Interpretation:**
- Mean return of 9.684 ± 1.030 means average episode reward is 9.684 with moderate variance
- Final 25% mean of 10.822 shows improvement from early episodes

---

### Episode Length (Turns per Episode)
**What it measures:** Number of dialogue turns before episode termination.

**How it's computed:** Count of turns until `done=True` or max turns reached

**What's good:**
- **Not hitting max:** Episodes should conclude naturally (not always hit max turns)
- **Reasonable length:** 8-15 turns for museum tours (not too short, not too long)
- **Stable:** Mean length should stabilize as agent learns

**Interpretation:**
- Mean length of 10.0 turns indicates average conversation duration
- If always hitting max (e.g., 40 turns), agent isn't learning to conclude properly
- Too short (<5 turns) suggests premature termination or poor engagement

---

## Episode-Level Metrics

### Episode Coverage
**What it measures:** Percentage of exhibits visited (with at least 1 fact mentioned).

**How it's computed:** $\text{coverage} = \frac{|\{e: |F_e^{used}| > 0\}|}{|\text{all exhibits}|} \times 100\%$

**What's good:**
- **Increasing trend:** Should improve over training
- **Target:** 20-40% coverage per episode (depends on museum size)
- **Diversity:** Higher coverage = more diverse tours

**Interpretation:**
- Coverage of 22.7% means agent covered ~2.3 exhibits out of 10 (example)
- Low coverage (<10%) suggests agent is stuck on one exhibit
- High coverage (>50%) might indicate shallow coverage (few facts per exhibit)

---

### Episode Facts
**What it measures:** Total number of new facts presented during an episode.

**How it's computed:** Count of unique fact IDs mentioned during episode

**What's good:**
- **Target range:** 5-10 facts per episode (informative but not overwhelming)
- **Increasing trend:** Should improve as agent learns to share more content
- **Stability:** Consistent fact delivery across episodes

**Interpretation:**
- 5.7 facts/episode means agent is providing substantive information
- Too few (<3) suggests agent isn't sharing enough content
- Too many (>15) might overwhelm users or indicate redundancy

---

## Turn-Level Metrics

### Per-Turn Reward
**What it measures:** Reward received at a single turn.

**How it's computed:** $r_t = r_t^{eng} + r_t^{nov} + r_t^{resp} + r_t^{trans} + r_t^{conclude}$

**What's good:**
- **Positive values:** Most turns should have positive reward
- **Variability:** Some variance is expected (not all turns are equal)
- **Trend:** Should increase over training as agent improves

**Interpretation:**
- High per-turn reward = agent made good choices (answered questions, shared facts, etc.)
- Negative per-turn reward = agent made mistakes (deflected, premature transition, etc.)

---

### Dwell Time
**What it measures:** User engagement level (gaze/attention) at current turn.

**How it's computed:** Normalized gaze feature from simulator: $\text{dwell}_t \in [0, 1]$

**What's good:**
- **Target:** Mean dwell > 0.4 indicates good engagement
- **Increasing trend:** Should improve as agent learns to engage users
- **Stability:** Consistent dwell suggests reliable engagement

**Interpretation:**
- Dwell of 0.48 means user is moderately engaged (48% of max attention)
- Low dwell (<0.3) suggests user is disengaged or confused
- High dwell (>0.7) indicates strong interest and engagement

---

## Reward Component Metrics

### Engagement Reward ($r^{eng}$)
**What it measures:** Reward from user engagement (dwell time).

**How it's computed:** $r_t^{eng} = \text{dwell}_t \times w_{eng}$ where $w_{eng} = 1.0$

**What's good:**
- **Positive values:** Should be positive most turns (dwell > 0)
- **Mean per episode:** 2-5 points per episode (depends on episode length)
- **Contribution:** Typically 40-60% of total reward

**Interpretation:**
- High engagement reward = user is paying attention and engaged
- Low engagement reward = user is disengaged or confused
- Increasing trend = agent is learning to maintain engagement

---

### Novelty Reward ($r^{nov}$)
**What it measures:** Reward for presenting new facts (not previously mentioned).

**How it's computed:** $r_t^{nov} = \alpha \times |\text{new facts at } t|$ where $\alpha = 0.15$ (baseline) or configurable

**What's good:**
- **Positive values:** Should be positive when new facts are shared
- **Mean per episode:** 2-6 points per episode (depends on facts shared)
- **Contribution:** Typically 30-60% of total reward

**Interpretation:**
- High novelty reward = agent is sharing new information
- Low novelty reward = agent is repeating facts or not sharing content
- Stable signal = agent is consistently providing novel information

---

### Responsiveness Reward ($r^{resp}$)
**What it measures:** Reward for answering user questions vs deflecting.

**How it's computed:**
$$r_t^{resp} = \begin{cases}
+0.25 & \text{if user asked at } t-1 \text{ \& agent adds new fact(s) at } t \\
-0.15 & \text{if user asked at } t-1 \text{ \& agent uses AskQuestion at } t \\
0 & \text{otherwise}
\end{cases}$$

**What's good:**
- **Positive values:** Should be positive when answering questions
- **Mean per episode:** 0.0 to +0.5 (depends on question frequency)
- **Contribution:** Typically 0-10% of total reward (small but important)

**Interpretation:**
- Positive responsiveness = agent answered user's question
- Negative responsiveness = agent deflected with counter-question
- Increasing trend = agent is learning to answer instead of deflect

---

### Transition Reward ($r^{trans}$)
**What it measures:** Penalty for premature transitions (offering moves too early).

**How it's computed:**
$$r_t^{trans} = \begin{cases}
-0.20 & |F_{current}| = 0 \\
-0.16 & |F_{current}| = 1 \\
0 & |F_{current}| \geq 2 \text{ or within 3-turn post-success exemption}
\end{cases}$$

**What's good:**
- **Zero or positive:** Should be 0 most turns (agent waits for sufficient facts)
- **Mean per episode:** -0.2 to 0.0 (negative indicates premature transitions)
- **Contribution:** Typically -2% to 0% of total reward

**Interpretation:**
- Negative transition reward = agent offered transition too early
- Zero transition reward = agent waited for sufficient facts (good)
- Decreasing negative trend = agent is learning to wait before transitioning

---

### Conclude Reward ($r^{conclude}$)
**What it measures:** Bonus for covering multiple exhibits when concluding.

**How it's computed:** $r_t^{conclude} = 0.2 \times |\{e: |F_e^{used}| > 0\}|$ (only on Conclude action)

**What's good:**
- **Positive values:** Should be positive when concluding with good coverage
- **Mean per episode:** 0.0 to +1.0 (depends on coverage and conclude usage)
- **Contribution:** Typically 0-10% of total reward

**Interpretation:**
- High conclude reward = agent concluded with good breadth (multiple exhibits)
- Zero conclude reward = agent didn't conclude or concluded with narrow coverage
- Increasing trend = agent is learning to conclude with better coverage

---

## Hierarchical Control Metrics

### Option Usage Distribution
**What it measures:** Frequency of each high-level option (Explain, AskQuestion, OfferTransition, Conclude).

**How it's computed:** Count of each option divided by total actions

**What's good:**
- **Balanced:** No single option should dominate (>70%)
- **Diversity:** All 4 options should be used (option diversity = 4)
- **Stability:** Usage patterns should stabilize over training

**Interpretation:**
- Explain 33.3%, AskQuestion 43.3%, OfferTransition 23.3% shows balanced usage
- One option >70% suggests agent is over-relying on one strategy
- Option diversity <3 suggests agent isn't using full action space

---

### Option Persistence (Duration)
**What it measures:** How many turns agent stays in the same option before switching.

**How it's computed:** Mean turns per option instance before termination or switch

**What's good:**
- **Target:** Mean persistence > 2 turns (shows coherent multi-turn strategies)
- **Distribution:** Should have some long-duration options (3-5 turns)
- **Stability:** Persistence should increase as agent learns

**Interpretation:**
- Mean persistence of 2.5 turns means agent typically uses an option for 2-3 turns
- Persistence <1.5 suggests agent is switching too frequently (not coherent)
- Persistence >4 might indicate agent is stuck in one option

---

### Option Transitions
**What it measures:** Patterns of switching between options (from → to).

**How it's computed:** Count of transitions from each option to each other option

**What's good:**
- **Natural patterns:** Common transitions should make sense (e.g., Explain → OfferTransition)
- **No cycles:** Shouldn't see rapid back-and-forth (e.g., Explain ↔ AskQuestion)
- **Stability:** Transition patterns should stabilize over training

**Interpretation:**
- Explain → OfferTransition is natural (explain, then suggest move)
- AskQuestion → AskQuestion repeatedly suggests deflection problem
- Rapid cycling indicates lack of coherent strategy

---

## Dialogue Quality Metrics

### Question Answer Rate
**What it measures:** Percentage of user questions that agent answers (vs deflects).

**How it's computed:** $\text{answer rate} = \frac{\text{question\_answers}}{\text{question\_answers} + \text{question\_deflections}}$

**What's good:**
- **Target:** >75% answer rate (agent answers most questions)
- **Increasing trend:** Should improve as agent learns
- **Stability:** Consistent answer rate across episodes

**Interpretation:**
- Answer rate of 82% means agent answers 82% of user questions
- Low answer rate (<60%) suggests deflection problem
- High answer rate (>90%) indicates good responsiveness

---

### Deflection Rate
**What it measures:** Percentage of user questions that agent deflects with counter-questions.

**How it's computed:** $\text{deflection rate} = \frac{\text{question\_deflections}}{\text{question\_answers} + \text{question\_deflections}}$

**What's good:**
- **Target:** <25% deflection rate (agent rarely deflects)
- **Decreasing trend:** Should decrease as agent learns
- **Stability:** Low deflection rate indicates consistent answering

**Interpretation:**
- Deflection rate of 18% means agent deflects 18% of questions (acceptable)
- High deflection rate (>40%) indicates serious responsiveness problem
- Low deflection rate (<10%) indicates good question-answering behavior

---

### Transition Success Rate
**What it measures:** Percentage of transition offers that are accepted by the user.

**How it's computed:** $\text{success rate} = \frac{\text{transition\_successes}}{\text{transition\_attempts}}$

**What's good:**
- **Target:** >60% success rate (most transitions accepted)
- **Increasing trend:** Should improve as agent learns timing
- **Stability:** Consistent success rate indicates good transition timing

**Interpretation:**
- Success rate of 68% means 68% of transition offers are accepted
- Low success rate (<40%) suggests premature transitions (user rejects)
- High success rate (>80%) indicates good transition timing

---

### Facts Before Transition
**What it measures:** Average number of facts shared at current exhibit before offering transition.

**How it's computed:** Mean facts shared at exhibit when OfferTransition is used

**What's good:**
- **Target:** ≥2 facts before transition (sufficient coverage)
- **Increasing trend:** Should increase as agent learns to wait
- **Distribution:** Most transitions should occur after 2+ facts

**Interpretation:**
- Facts before transition of 2.3 means agent typically shares 2-3 facts before moving
- Low value (<1.5) suggests premature transitions
- High value (>4) might indicate agent is staying too long (but acceptable)

---

## Training Dynamics Metrics

### Learning Curve
**What it measures:** Episode returns over training episodes (with smoothing).

**How it's computed:** Moving average of episode returns: $\bar{R}_t = \frac{1}{W} \sum_{i=t-W+1}^{t} R_i$

**What's good:**
- **Increasing trend:** Should show clear upward trend
- **Smoothness:** Should be relatively smooth (not too noisy)
- **Convergence:** Should plateau at high performance (not decrease)

**Interpretation:**
- Upward trend = agent is learning
- Plateau = agent has converged to a policy
- Decreasing trend = agent is unlearning (rare, indicates problem)

---

### Convergence Rate
**What it measures:** Speed of improvement (early vs late performance).

**How it's computed:** $\text{improvement} = \bar{R}_{\text{late}} - \bar{R}_{\text{early}}$ where late = final 25%, early = first 25%

**What's good:**
- **Positive improvement:** Late performance should exceed early performance
- **Large improvement:** >2-3 points improvement shows significant learning
- **Fast convergence:** Improvement should occur within first 100-200 episodes

**Interpretation:**
- Improvement of +2.5 means agent improved by 2.5 points from early to late
- Large improvement (>5 points) indicates strong learning
- Small improvement (<1 point) suggests slow or limited learning

---

### Reward Variance
**What it measures:** Consistency of rewards (standard deviation).

**How it's computed:** $\sigma_R = \sqrt{\frac{1}{N-1} \sum_{i=1}^{N} (R_i - \bar{R})^2}$

**What's good:**
- **Low variance:** Lower variance indicates more consistent performance
- **Decreasing trend:** Variance should decrease as agent learns
- **Stability:** Stable variance indicates reliable learning

**Interpretation:**
- Variance of 1.030 means returns vary by ±1.030 around mean
- High variance (>2.0) indicates unstable learning
- Low variance (<0.5) indicates very consistent performance

---

## Success Rate Metrics

### Transition Attempts
**What it measures:** Total number of OfferTransition actions taken.

**How it's computed:** Count of OfferTransition actions across all episodes

**What's good:**
- **Reasonable frequency:** 15-30% of actions should be transitions
- **Not too frequent:** <50% (not spamming transitions)
- **Not too rare:** >5% (agent should offer transitions)

**Interpretation:**
- 23.3% transition usage means ~1 in 4 actions is a transition offer
- Too frequent (>40%) suggests transition spam
- Too rare (<5%) suggests agent isn't exploring museum

---

### Transition Successes
**What it measures:** Number of successful transitions (accepted by user).

**How it's computed:** Count of OfferTransition actions where user accepted

**What's good:**
- **High success rate:** >60% of transitions should succeed
- **Increasing trend:** Success rate should improve over training
- **Stability:** Consistent success rate indicates good timing

**Interpretation:**
- 68% success rate means 68% of transition offers are accepted
- Low success rate (<40%) indicates timing problems
- High success rate (>80%) indicates excellent timing

---

## Content Quality Metrics

### Hallucination Rate
**What it measures:** Percentage of facts that are hallucinated (not in KB).

**How it's computed:** $\text{hallucination rate} = \frac{\text{hallucinated facts}}{\text{total facts}} \times 100\%$

**What's good:**
- **Target:** <5% hallucination rate (very low)
- **Decreasing trend:** Should decrease as agent learns grounding
- **Stability:** Low and stable rate indicates good grounding

**Interpretation:**
- Hallucination rate of 2% means 2% of facts are invalid (acceptable)
- High rate (>10%) indicates serious grounding problem
- Zero rate is ideal but may be unrealistic

---

### KB-Grounding Precision
**What it measures:** Percentage of facts that are valid (in KB).

**How it's computed:** $\text{precision} = \frac{\text{valid facts}}{\text{valid facts} + \text{hallucinated facts}} \times 100\%$

**What's good:**
- **Target:** >95% precision (very high)
- **Increasing trend:** Should improve as agent learns
- **Stability:** High and stable precision indicates good grounding

**Interpretation:**
- Precision of 98% means 98% of facts are valid (excellent)
- Low precision (<90%) indicates grounding problems
- High precision (>98%) indicates excellent knowledge grounding

---

### Novelty Signal Stability
**What it measures:** Consistency of novelty reward (variance).

**How it's computed:** Standard deviation of novelty reward component

**What's good:**
- **Low variance:** Lower variance indicates more stable signal
- **Decreasing trend:** Should decrease as agent learns
- **Stability:** Stable variance indicates reliable novelty detection

**Interpretation:**
- Variance of 0.28 means novelty reward is relatively stable
- High variance (>0.5) indicates unstable novelty detection
- Low variance (<0.2) indicates very stable novelty signal

---

## Interpretation Guidelines

### What Makes a "Good" Training Run?

1. **Learning:** Episode returns increase over time (clear upward trend)
2. **Convergence:** Returns plateau at high value (not decreasing)
3. **Stability:** Low variance in returns (consistent performance)
4. **Coverage:** 20-40% exhibit coverage per episode
5. **Content:** 5-10 facts per episode
6. **Responsiveness:** >75% question answer rate
7. **Transitions:** >60% transition success rate, ≥2 facts before transition
8. **Grounding:** <5% hallucination rate, >95% KB precision
9. **Coherence:** Option persistence >2 turns, low invalid action rate
10. **Efficiency:** Episodes conclude naturally (not always hitting max)

### Red Flags (Indicators of Problems)

1. **No Learning:** Returns not increasing or decreasing
2. **High Variance:** Returns very unstable (std >2.0)
3. **Low Coverage:** <10% exhibit coverage
4. **High Deflection:** >40% deflection rate
5. **Low Transition Success:** <40% success rate
6. **High Hallucination:** >10% hallucination rate
7. **Low Persistence:** Option persistence <1.5 turns
8. **Always Max Length:** Episodes always hit max turns
9. **Invalid Actions:** >5% invalid action attempts
10. **Poor Grounding:** <90% KB precision

### Comparing Experiments

When comparing two experiments:

1. **Learning Curves:** Compare final performance and convergence speed
2. **Coverage:** Compare mean exhibit coverage
3. **Responsiveness:** Compare question answer rates
4. **Transitions:** Compare success rates and facts-before-transition
5. **Grounding:** Compare hallucination rates and KB precision
6. **Stability:** Compare reward variance
7. **Efficiency:** Compare training time and sample efficiency

### Statistical Significance

For reliable conclusions:
- **Minimum episodes:** 50-100 episodes for stable statistics
- **Multiple runs:** 3-5 runs with different seeds for robustness
- **Confidence intervals:** Report mean ± std or use confidence intervals
- **Trend analysis:** Use moving averages to identify trends
- **Hypothesis testing:** Use statistical tests for significance

---

## Summary

This guide covers all metrics tracked in the HRL Museum Agent system. Use these metrics to:

1. **Monitor training:** Track learning progress and identify problems
2. **Evaluate performance:** Assess agent capabilities and quality
3. **Compare experiments:** Identify which changes improve performance
4. **Debug issues:** Pinpoint specific problems (deflection, premature transitions, etc.)
5. **Report results:** Present quantitative evidence of improvements

Remember: Metrics are tools for understanding, not goals in themselves. Always interpret metrics in context of the overall system behavior and user experience.

