# HRL Museum Agent: Design Changes Driven by Testing
## Slide Notes for Presentation

**Goal:** Present the concrete problems we found in testing and the fixes we shipped in the RL formalization and simulator.

**Scope:** All new reward components beyond novelty & engagement, plus non-dwell simulator edits.

**TL;DR:** Each change solves a failure mode we observed (deflection, premature moves, narrow tours, etc.) with precise mechanisms + metrics.

---

## Baseline (for context only, 1 slide)

**Hierarchy:** Options over SMDP (Explain, AskQuestion, OfferTransition, Conclude) with learned termination.

**Compact state:** focus vector, brief history, 64-d intent, 64-d short context.

**Unchanged rewards:**
- $r_t^{eng} = \text{dwell}_t \in [0,1]$ (baseline)
- $r_t^{nov} = \alpha \cdot \Delta|F_{used}|, \alpha = 0.15$ (baseline)

---

## What Testing Revealed (Summary of Issues)

Early testing revealed several critical failure modes:

1. **Deflection:** When users asked questions, the agent frequently responded with counter-questions instead of providing answers. This led to lower engagement and user frustration.

2. **Premature Transitions:** The agent would offer to move to new exhibits after sharing very few facts (often 0-1 facts), which users rejected, causing confusion and resulting in shallow coverage of exhibits.

3. **Narrow Tours:** Sessions often concluded with the agent having covered only 1-2 exhibits, missing opportunities for broader museum exploration.

4. **Degenerate Actions:** The agent exhibited pathological behaviors: concluding immediately, attempting to repeat facts before any had been shared, or trying to share new facts when none remained available.

5. **Gaming the System:** The agent could exploit the simulator by earning novelty credit for hallucinated facts that weren't actually in the knowledge base.

6. **Training Efficiency:** LLM-only simulation was too slow for rapid iteration, while template-only modes lacked the diversity needed for realistic evaluation.

---

## Change 1 — Responsiveness Reward: Stop Deflecting

**Problem (from early testing):** After a user asked a question, the agent often replied with a counter-question instead of answering. This deflection pattern led to lower dwell times and user frustration.

**Design choice:** Explicitly reward answering user questions and penalize deflection with counter-questions.

**Mechanism:**
$$r_t^{resp} = \begin{cases}
+0.25 & \text{if user asked at } t-1 \text{ \& agent adds new fact(s) at } t \\
-0.15 & \text{if user asked at } t-1 \text{ \& agent uses AskQuestion at } t \\
0 & \text{otherwise}
\end{cases}$$

**What we track:** Deflection rate (percentage of questions deflected), answer-after-question ratio, and dwell time on question-answer turns.

**Early testing results:** The responsiveness reward successfully reduced deflection behavior. Agents learned to answer questions more frequently, leading to improved engagement on question-answer turns. The reward signal guides the agent away from deflection patterns and toward more helpful responses.

---

## Change 2 — Transition Insufficiency Penalty: Don't Leave Too Soon

**Problem (from early testing):** The agent proposed transitions to new exhibits after sharing very few facts (0-1 facts), which users consistently rejected. This created confusion and resulted in shallow coverage of exhibits.

**Design choice:** Penalize early transitions while allowing flexibility after a recent successful transition.

**Mechanism:**
$$r_t^{trans} = \begin{cases}
-0.20 & |F_{current}| = 0 \\
-0.16 & |F_{current}| = 1 \\
0 & |F_{current}| \geq 2 \text{ or within 3-turn post-success exemption}
\end{cases}$$

**What we track:** Average facts shared before transitions, transition acceptance rate, and frequency of back-to-back failed transitions.

**Early testing results:** The transition penalty encourages the agent to wait until it has shared sufficient facts (typically 2+) before offering to move. This leads to higher acceptance rates and reduces the confusion caused by premature transition attempts. The 3-turn exemption rule provides flexibility after successful transitions.

---

## Change 3 — Conclude Bonus: Favor Breadth at the End

**Problem (from early testing):** Agents would over-focus on a single exhibit and conclude sessions early, resulting in narrow tours that missed opportunities for broader exploration.

**Design choice:** Reward breadth when concluding by paying a bonus proportional to the number of exhibits covered.

**Mechanism (only on conclude):**
$$r_t^{conclude} = 0.2 \times |\{e: |F_e^{used}| > 0\}|$$

**What we track:** Number of exhibits with at least one fact at episode end, coverage diversity percentage, and conclude option usage frequency.

**Early testing results:** The conclude bonus incentivizes the agent to explore multiple exhibits before concluding. While the conclude option itself may still be used infrequently, the reward structure encourages broader coverage throughout the session, leading to more diverse tours.

---

## Change 4 — Action Masking: Prevent Degenerate Moves

**Problem (from early testing):** The agent exhibited pathological behaviors: concluding immediately without context, attempting to repeat facts before any had been shared, or trying to share new facts when none remained available.

**Design choice:** Hard action masks that prevent structurally incoherent actions.

**Mechanism:**
- Mask `Conclude` until ≥3 total facts and ≥2 exhibits touched.
- Mask `ExplainNewFact` when no new facts remain for current exhibit.
- Mask `RepeatFact` if no fact has been mentioned yet.
- `ClarifyFact` always allowed.

**What we track:** Invalid action attempt rate (should be zero), coherent span length (turns per option), and option switch rate.

**Early testing results:** Action masking completely eliminates degenerate action attempts. The agent maintains longer coherent spans within each option, leading to smoother learning curves and more natural dialogue patterns. This structural constraint helps the agent learn more effectively by preventing it from exploring impossible action sequences.

---

## Change 5 — Transition Acceptance Model: Teach Timing

**Problem (from early testing):** The simulator accepted transition offers too easily, allowing the agent to spam transitions without learning proper timing. The agent didn't learn when transitions were appropriate.

**Design choice:** Make transition acceptance probability depend on local coverage (facts shared at current exhibit).

**Mechanism:**
$$p_{accept} = \begin{cases}
0.20 & 0 \text{ facts} \\
0.50 & 1 \text{ fact} \\
0.80 & 2-3 \text{ facts} \\
0.95 & 4+ \text{ facts}
\end{cases}$$

**What we track:** Transition acceptance rate, number of retries per episode, and distribution of facts-before-transition for successful transitions.

**Early testing results:** The probability-based acceptance model teaches the agent that transitions are more likely to succeed when sufficient facts have been shared. This leads to higher acceptance rates overall, fewer retries, and transitions that naturally cluster after 2-3 facts have been presented. The agent learns proper timing through experience rather than hard rules.

---

## Change 6 — Engagement Memory (Patience/Decay): Don't Ignore Streaks

**Problem (from early testing):** When the agent made two bad turns in a row (mismatch or deflection), the cumulative impact on returns was minimal. The agent didn't learn from error streaks because the signal was too weak.

**Design choice:** Decay subsequent engagement signals after consecutive poor turns to amplify the cost of error streaks.

**Mechanism:** After two consecutive mismatch/deflection events, reduce future dwell-derived signals by 40% until the agent recovers with a successful turn.

**What we track:** Maximum consecutive error streak length per episode, recovery time (turns to return to baseline), and early exit rate.

**Early testing results:** The engagement decay mechanism makes error streaks more costly, encouraging the agent to recover quickly. This leads to shorter error streaks, faster recovery behavior, and fewer early episode terminations. The agent learns to avoid consecutive mistakes and maintain engagement more consistently.

---

## Change 7 — Dual Generation Modes: Keep Training Fast, Eval Realistic

**Problem (from early testing):** Using LLM-only simulation for every turn was too slow for rapid training iteration, while template-only modes lacked the diversity and context-awareness needed for realistic evaluation.

**Design choice:** Use two generation modes: fast template mode for training throughput, and LLM mode for diverse, context-aware evaluation.

**Mechanism:** Template mode generates responses quickly during training for high throughput. LLM mode provides diverse, context-aware responses for evaluation and spot checks to ensure training quality.

**What we track:** Training time per episode, simulator LLM call time, response diversity, and performance comparison between modes.

**Early testing results:** Dual generation modes significantly reduce training time while maintaining evaluation quality. Template mode provides 2-3x faster training throughput, while LLM mode ensures we can still evaluate with realistic, diverse responses. This allows for faster iteration cycles without sacrificing evaluation rigor.

---

## Change 8 — Fact Verification & Novelty Gating: No Free Credit

**Problem (from early testing):** The agent sometimes earned novelty credit for hallucinated facts—facts that weren't actually in the knowledge base. This corrupted the learning signal and allowed the agent to game the reward system.

**Design choice:** Validate all fact IDs against the current exhibit's knowledge base before awarding novelty credit.

**Mechanism:** Extract fact IDs from agent utterances → validate against KB → classify as (new/valid, repeat, or hallucinated) → only new/valid facts count toward $r^{nov}$.

**What we track:** Hallucination rate (percentage of facts that are invalid), KB-grounding precision, and stability of the novelty reward signal.

**Early testing results:** Fact verification dramatically reduces hallucination rates and improves knowledge base grounding precision. The novelty reward signal becomes more stable and reliable, providing cleaner learning signals. The agent can no longer earn rewards for made-up facts, forcing it to ground its responses in the actual knowledge base.

---

## All Rewards Together (What's New vs Baseline)

**Total per turn:**
$$R_t = \underbrace{r_t^{eng}}_{\text{baseline}} + \underbrace{r_t^{nov}}_{\text{baseline}} + r_t^{resp} + r_t^{trans} + r_t^{conclude}$$

**New components:** $r_t^{resp}$ (anti-deflection), $r_t^{trans}$ (anti-premature move), $r_t^{conclude}$ (breadth).

**Note:** Dwell definition itself is unchanged; decay is a simulator-side modulation of its effect, not its measure.

**Early testing observations:** The new reward components work together to guide the agent toward more helpful behaviors. Engagement and novelty remain the primary drivers, while the new components provide fine-grained guidance for specific failure modes. The reward decomposition shows that responsiveness and transition penalties are small but important signals that shape behavior.

---

## What Did NOT Change (for clarity)

- Dwell ratio computation and windowing (baseline).
- Novelty increment form and $\alpha = 0.15$ (baseline).
- Options/SMDP architecture, actor–critic with learned termination.
- Simulator personas and response-type classification (already existed).

---

## How We'll Know It Worked (KPI Shifts Aligned to Each Fix)

### Responsiveness
- **Deflection rate:** Should decrease significantly as agent learns to answer questions.
- **Answer-after-question ratio:** Should increase, showing more helpful responses.
- **Dwell on Q→A turns:** Should improve, indicating better engagement when answering.

### Transitions
- **Facts-before-transition:** Should increase, showing agent waits for sufficient coverage.
- **Acceptance rate:** Should improve as transitions occur at better times.
- **Back-to-back failed transitions:** Should decrease, reducing confusion.

### Breadth
- **Exhibits with ≥1 fact at end:** Should increase, showing broader exploration.
- **Coverage diversity:** Should improve, indicating more diverse tours.

### Coherence
- **Coherent span length:** Should increase, showing longer option persistence.
- **Invalid action attempts:** Should be zero (masked actions prevented).
- **Option switch rate:** Should decrease, indicating more stable strategies.

### Grounding
- **Hallucination rate:** Should decrease dramatically.
- **KB-citation precision:** Should improve, showing better grounding.
- **Novelty signal stability:** Should improve, providing cleaner learning signals.

### Learning Stability
- **Convergence rate:** Should show faster improvement in early episodes.
- **Reward variance:** Should decrease, indicating more stable learning.
- **Training time:** Should decrease with dual generation modes.

---

## Modern RL Evaluation Practices

We track metrics aligned with modern RL evaluation practices:

**Core RL Metrics:**
- Learning curves (episode returns with moving averages)
- Sample efficiency (episodes to reach target performance)
- Convergence rate (early vs late performance)
- Stability (variance across episodes and seeds)

**Hierarchical RL Metrics:**
- Option persistence (mean turns per option)
- Option diversity (unique options used)
- Option transitions (switching patterns)
- Termination learning effectiveness

**Task-Specific Metrics:**
- Coverage (exhibits/facts per episode)
- Responsiveness (answer rate after questions)
- Transition timing (facts before transitions)
- Grounding (KB precision, hallucination rate)

**Reward Analysis:**
- Reward decomposition (component contributions)
- Reward evolution (how components change)
- Signal-to-noise (variance and consistency)

These metrics help us understand both the learning dynamics and the quality of the learned policy, providing a comprehensive view of agent performance.
