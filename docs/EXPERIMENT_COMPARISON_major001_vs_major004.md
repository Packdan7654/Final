# Experiment Comparison: major_001 vs major_004

## Overview

| Metric | major_001 | major_004 | Difference |
|--------|-----------|-----------|------------|
| **Episodes** | 1000 | 300 | major_001: 3.3x longer |
| **Training Duration** | ~8.4 hours | 2.4 hours | major_001: 3.5x longer |
| **Learning Rate** | 0.0003 | 0.0001 | major_001: 3x higher |
| **Status** | Completed | Completed | Both completed |

---

## Key Configuration Differences

### Reward Parameters

| Parameter | major_001 | major_004 | Impact |
|-----------|-----------|-----------|--------|
| **Novelty per fact** | 0.25 | 1.0 | **4x higher in major_004** - Strongly incentivizes sharing facts |
| **Conclude bonus** | 0.2 | 0.5 | **2.5x higher in major_004** - More reward for breadth |
| **Responsiveness** | 0.25 | 0.25 | Same |
| **Transition penalty** | -0.2 | -0.2 | Same |
| **Engagement** | 1.0 | 1.0 | Same |

**Key Insight:** major_004 uses much higher novelty reward (1.0 vs 0.25) and conclude bonus (0.5 vs 0.2), which significantly changes the reward landscape.

---

## Learning Performance

### Episode Returns

| Metric | major_001 | major_004 | Analysis |
|--------|-----------|-----------|----------|
| **Mean Return** | 13.922 ± 11.782 | 29.611 ± 9.909 | **major_004: 2.1x higher** |
| **Final 25% Mean** | 12.460 | 32.671 | **major_004: 2.6x higher** |
| **Improvement (early→late)** | **-6.180** ⚠️ | **+15.109** ✅ | **Critical difference!** |
| **Variance (std)** | 11.782 | 9.909 | major_004: More stable |

**Key Findings:**
- **major_001 shows negative learning:** Returns decreased from early to late training (-6.18 points)
- **major_004 shows strong positive learning:** Returns increased by +15.11 points
- **major_004 achieves 2x higher returns** despite training for 1/3 the episodes
- **major_004 is more stable** (lower variance)

### Episode Length

| Metric | major_001 | major_004 | Analysis |
|--------|-----------|-----------|----------|
| **Mean Episode Length** | 39.5 turns | 35.3 turns | major_004: 4.2 turns shorter |
| **Max Turns** | 40 | 40 | Same |

**Key Finding:** major_004 episodes are shorter, suggesting better efficiency and possibly better conclusion behavior (though conclude usage is still low).

---

## Content Quality (RQ3)

### Coverage & Facts

| Metric | major_001 | major_004 | Improvement |
|--------|-----------|-----------|------------|
| **Mean Exhibit Coverage** | 18.5% | 35.8% | **+93% improvement** |
| **Mean Facts/Episode** | 4.6 | 9.0 | **+96% improvement** |

**Key Findings:**
- **major_004 covers nearly 2x more exhibits** (35.8% vs 18.5%)
- **major_004 shares nearly 2x more facts** (9.0 vs 4.6 per episode)
- The higher novelty reward (1.0 vs 0.25) strongly incentivizes fact-sharing
- Better coverage suggests the conclude bonus (0.5 vs 0.2) is working

---

## Hierarchical Control (RQ1)

### Option Usage Distribution

| Option | major_001 | major_004 | Analysis |
|--------|-----------|-----------|----------|
| **Explain** | 44.2% | **80.8%** | major_004: 1.8x more frequent |
| **AskQuestion** | 35.5% | **10.0%** | major_001: 3.5x more frequent |
| **OfferTransition** | 20.2% | 8.5% | major_001: 2.4x more frequent |
| **Conclude** | 0.1% | 0.7% | major_004: 7x more frequent (still very low) |
| **Option Diversity** | 4 | 4 | Same (all options used) |

**Key Findings:**
- **major_004 heavily favors Explain** (80.8% vs 44.2%) - likely due to high novelty reward
- **major_001 has more balanced usage** across Explain, AskQuestion, and OfferTransition
- **major_004 uses AskQuestion much less** (10.0% vs 35.5%) - possibly because it's focused on sharing facts
- **Conclude usage is still very low in both** (0.1% vs 0.7%), but major_004 is 7x better

**Interpretation:**
- Higher novelty reward in major_004 makes Explain (fact-sharing) much more attractive
- This creates a more focused strategy: explain facts rather than ask questions or transition
- The strategy is effective (higher returns, better coverage) but less diverse

---

## Reward Decomposition

### Average Reward Components per Episode

| Component | major_001 | major_004 | Analysis |
|-----------|-----------|-----------|----------|
| **Engagement** | N/A | 20.741 (70.0%) | Major driver in major_004 |
| **Novelty** | N/A | 8.953 (30.2%) | Significant in major_004 |
| **Conclude** | N/A | 0.303 (1.0%) | Small but present |
| **Transition** | N/A | -0.285 (-1.0%) | Small penalty |
| **Responsiveness** | N/A | -0.101 (-0.3%) | Small negative |
| **Total** | 13.922 | 29.611 | major_004: 2.1x higher |

**Key Findings:**
- **Engagement is the primary driver** (70% of total reward in major_004)
- **Novelty contributes significantly** (30% of total reward)
- Other components are small but present
- The higher novelty reward scale (1.0 vs 0.25) directly increases novelty contribution

---

## Learning Dynamics Analysis

### Learning Trajectory

**major_001:**
- ❌ **Negative learning:** Returns decreased from early to late training
- ⚠️ **High variance:** std = 11.782 (unstable)
- 📉 **Declining performance:** Final 25% mean (12.46) < Overall mean (13.92)

**major_004:**
- ✅ **Positive learning:** Returns increased by +15.11 points
- ✅ **Lower variance:** std = 9.909 (more stable)
- 📈 **Improving performance:** Final 25% mean (32.67) > Overall mean (29.61)

**Interpretation:**
- major_001's higher learning rate (0.0003) may have caused instability or overfitting
- major_004's lower learning rate (0.0001) with higher novelty reward creates stable, improving learning
- The reward structure in major_004 (higher novelty, higher conclude bonus) better aligns with desired behavior

---

## Key Takeaways

### What Worked Better in major_004:

1. **✅ Much higher returns** (29.6 vs 13.9) - 2.1x improvement
2. **✅ Positive learning trajectory** (+15.1 improvement vs -6.2 decline)
3. **✅ Better coverage** (35.8% vs 18.5%) - nearly 2x
4. **✅ More facts shared** (9.0 vs 4.6 per episode) - nearly 2x
5. **✅ More stable learning** (lower variance)
6. **✅ Better conclude usage** (0.7% vs 0.1%) - still low but 7x better

### Trade-offs:

1. **⚠️ Less balanced option usage** - major_004 heavily favors Explain (80.8%)
2. **⚠️ Less question-asking** - AskQuestion usage dropped from 35.5% to 10.0%
3. **⚠️ Fewer transitions** - OfferTransition usage dropped from 20.2% to 8.5%

### Configuration Recommendations:

**For better performance (major_004 approach):**
- ✅ Use higher novelty reward (1.0 instead of 0.25)
- ✅ Use higher conclude bonus (0.5 instead of 0.2)
- ✅ Use lower learning rate (0.0001 instead of 0.0003) for stability

**For more balanced behavior (major_001 approach):**
- ⚠️ Lower novelty reward (0.25) creates more balanced option usage
- ⚠️ But this comes at the cost of lower returns and negative learning

---

## Conclusion

**major_004 significantly outperforms major_001** in almost every metric:
- 2.1x higher returns
- Positive learning vs negative learning
- 2x better coverage and fact-sharing
- More stable training

The key difference is the **reward structure**: higher novelty reward (1.0 vs 0.25) and conclude bonus (0.5 vs 0.2) create a stronger learning signal that leads to better performance.

However, this comes with a trade-off: major_004's strategy is less diverse (heavily favors Explain), while major_001 has more balanced option usage. Whether this is acceptable depends on the desired behavior - if the goal is maximum information delivery and coverage, major_004 is clearly superior.

