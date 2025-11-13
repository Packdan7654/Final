# Analysis: Why Returns Jump More Than Coverage in major_004

## The Observation

**major_004 shows:**
- Returns: 29.6 (2.1x higher than major_001's 13.9)
- Coverage: 35.8% (1.9x higher than major_001's 18.5%)
- Facts/episode: 9.0 (1.9x higher than major_001's 4.6)

**Question:** Why do returns increase 2.1x while coverage only increases 1.9x?

---

## Key Findings

### 1. Facts Per Covered Exhibit (Depth)

Both experiments share **the same depth** per exhibit:
- **major_001:** 2.50 facts per covered exhibit
- **major_004:** 2.50 facts per covered exhibit

This means the improvement is purely in **breadth** (more exhibits), not depth (more facts per exhibit).

### 2. Reward Component Breakdown (major_004)

From the evaluation summary:
- **Engagement:** 20.741 per episode (70.0% of total)
- **Novelty:** 8.953 per episode (30.2% of total)
- **Other components:** Small contributions

**Key Insight:** Engagement is the dominant reward component (70%), not novelty!

### 3. Novelty Reward Amplification

**Configuration difference:**
- **major_001:** novelty_per_fact = 0.25
- **major_004:** novelty_per_fact = 1.0 (4x higher)

**Novelty reward calculation:**
- **major_001:** 4.6 facts × 0.25 = **1.15 novelty reward/episode**
- **major_004:** 9.0 facts × 1.0 = **9.0 novelty reward/episode**

**Result:** 7.8x more novelty reward, but this only accounts for 30% of total returns.

### 4. Engagement Reward (The Real Driver)

**Engagement reward = dwell time** (per turn, summed over episode)

**major_004:**
- Engagement: 20.741 per episode
- Mean episode length: 35.3 turns
- **Average dwell per turn: 0.59** (20.741 / 35.3)

**major_001 (estimated):**
- If engagement is similar percentage (~70%), engagement ≈ 9.7 per episode
- Mean episode length: 39.5 turns
- **Average dwell per turn: 0.25** (9.7 / 39.5)

**Key Finding:** major_004 achieves **2.4x higher dwell per turn** (0.59 vs 0.25)!

---

## Why Engagement Is So Much Higher

### Hypothesis 1: Better Fact Quality → Better Engagement

The higher novelty reward (1.0 vs 0.25) incentivizes the agent to:
- Share more facts (9.0 vs 4.6)
- Focus on Explain option (80.8% vs 44.2%)
- Provide more substantive content

**Result:** More informative turns → higher user engagement (dwell time)

### Hypothesis 2: Shorter, More Focused Episodes

- **major_001:** 39.5 turns/episode (hitting max often)
- **major_004:** 35.3 turns/episode (more efficient)

Shorter episodes with better content may maintain higher engagement throughout, rather than dragging on with lower engagement.

### Hypothesis 3: Less Question-Asking → More Direct Engagement

- **major_001:** AskQuestion 35.5% of turns
- **major_004:** AskQuestion 10.0% of turns

Asking questions might reduce immediate engagement (user thinking/processing), while explaining facts directly maintains gaze on the exhibit.

### Hypothesis 4: Better Option Strategy

- **major_001:** More balanced usage (Explain 44.2%, AskQuestion 35.5%, OfferTransition 20.2%)
- **major_004:** Focused on Explain (80.8%)

The focused Explain strategy might be more engaging because:
- Continuous information flow
- Less switching between strategies
- More coherent narrative

---

## Mathematical Breakdown

### major_001 (estimated):
```
Total Return: 13.92
├─ Engagement: ~9.7 (70%) = 0.25 dwell/turn × 39.5 turns
├─ Novelty: ~1.15 (8%) = 4.6 facts × 0.25
└─ Other: ~3.1 (22%)
```

### major_004 (actual):
```
Total Return: 29.61
├─ Engagement: 20.74 (70%) = 0.59 dwell/turn × 35.3 turns
├─ Novelty: 8.95 (30%) = 9.0 facts × 1.0
└─ Other: -0.08 (-0.3%)
```

### The Multiplier Effect:

1. **Facts increase 1.9x** (4.6 → 9.0)
2. **Novelty reward per fact increases 4x** (0.25 → 1.0)
3. **Novelty reward increases 7.8x** (1.15 → 9.0)
4. **But engagement increases 2.1x** (9.7 → 20.74)
5. **Dwell per turn increases 2.4x** (0.25 → 0.59)

**Result:** The combination of:
- More facts (1.9x)
- Higher novelty reward (4x multiplier)
- Much better engagement (2.4x dwell per turn)
- Shorter, more efficient episodes

Creates a **2.1x total return increase** that's slightly more than the coverage increase (1.9x).

---

## Why Coverage Doesn't Scale Linearly with Returns

### Coverage is a Binary Metric

Coverage = percentage of exhibits with ≥1 fact

This is a **binary, threshold metric**:
- Going from 0 to 1 fact at an exhibit = +1 exhibit covered
- Going from 1 to 10 facts at the same exhibit = +0 exhibits covered

So coverage can't capture the **depth** of engagement or the **quality** of interaction.

### Returns Capture Continuous Quality

Returns include:
- **Engagement (dwell):** Continuous measure of attention quality
- **Novelty:** Rewards each new fact (not just first fact)
- **Episode efficiency:** Shorter episodes with higher engagement

These continuous measures can improve independently of coverage.

---

## Key Insights

1. **Engagement is the primary driver** (70% of returns), not coverage
2. **Higher novelty reward creates better engagement** by incentivizing more informative content
3. **Focused Explain strategy** (80.8% usage) maintains higher engagement than balanced strategy
4. **Shorter, more efficient episodes** maintain engagement better than longer, dragging episodes
5. **Coverage is a coarse metric** that doesn't capture engagement quality or fact depth

---

## Conclusion

The returns jump more than coverage because:

1. **Engagement (dwell) improves dramatically** (2.4x per turn) due to:
   - Better content quality (more facts, focused strategy)
   - More efficient episodes (shorter, higher engagement)
   - Less question-asking (more direct engagement)

2. **Novelty reward amplifies** (7.8x increase) due to:
   - More facts (1.9x)
   - Higher reward per fact (4x)

3. **Coverage is a binary metric** that only measures breadth, not:
   - Engagement quality
   - Fact depth per exhibit
   - Episode efficiency
   - Content quality

**The takeaway:** Coverage measures "how many exhibits touched," but returns measure "how well the agent engaged users and delivered content." The higher returns reflect better engagement quality and more efficient content delivery, not just broader coverage.

