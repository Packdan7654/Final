# Paper Analysis: Discrepancies and Structural Recommendations

## Executive Summary

This document identifies discrepancies between `paper.tex` and the codebase implementation, and provides structural recommendations to improve the paper's organization for a proper master's thesis format.

---

## Part 1: Discrepancies Between Paper and Codebase

### 1.1 Subaction Naming Inconsistency

**Issue**: Paper uses `AssessAndTransition` but codebase implements `SuggestMove`

**Location in Paper**:
- Line 455: Table lists `AssessAndTransition` for `OfferTransition` option
- Line 472: Text mentions `AssessAndTransition`
- Line 777: Prompt construction example uses `AssessAndTransition`

**Location in Code**:
- `src/environment/env.py:89`: `"OfferTransition": ["SuggestMove"]`
- `src/utils/dialogue_planner.py:48`: `if subaction == "SuggestMove":`

**Recommendation**: Either:
- Update paper to use `SuggestMove` throughout (simpler, matches code)
- OR update code to use `AssessAndTransition` (more descriptive)

**Priority**: HIGH - This is a direct naming mismatch that affects clarity.

---

### 1.2 Reward Function: Transition Spam Penalty

**Issue**: Paper describes transition spam as a reward component (`r^{spam}_t`), but code implements it at simulator level

**Location in Paper**:
- Line 601-606: Defines `r^{spam}_t = -0.10 × max(0, n_{consec} - 2)`
- Line 634: Includes `r^{spam}_t` in combined reward function

**Location in Code**:
- `src/environment/env.py:78, 313, 358`: Comments state "Transition spam is handled at simulator level (reduces dwell time)"
- `src/simulator/sim8_adapter.py:898-935`: Implements transition spam as a multiplier that reduces dwell time, not as a direct reward

**Recommendation**: 
- Update paper to clarify that transition spam is handled at simulator level (affects `dwell_t` which feeds into `r^{eng}_t`)
- Remove `r^{spam}_t` from the explicit reward equation or note it's implicit via engagement reduction
- The current implementation is actually more elegant (spam reduces engagement, which naturally reduces reward)

**Priority**: MEDIUM - Functional behavior is correct, but paper description doesn't match implementation.

---

### 1.3 State Space Dimensions

**Issue**: Paper states 149-d for 8 exhibits, but code calculates dynamically

**Location in Paper**:
- Line 412: "Total state dimension (8 exhibits): 9 (focus) + 12 (history) + 64 (intent) + 64 (context) = **149**"

**Location in Code**:
- `src/environment/env.py:115-120`: Dimensions calculated dynamically:
  - `focus_dim = n_exhibits + 1`
  - `history_dim = n_exhibits + len(options)` (4 options)
  - `intent_dim = 64`
  - `context_dim = 64`
  - Total = `(n_exhibits + 1) + (n_exhibits + 4) + 64 + 64`

**For 8 exhibits**: (8+1) + (8+4) + 64 + 64 = 9 + 12 + 64 + 64 = 149 ✓ (Matches!)
**For 5 exhibits**: (5+1) + (5+4) + 64 + 64 = 6 + 9 + 64 + 64 = 143

**Recommendation**: 
- Paper is correct for 8 exhibits, but add a note that dimensions scale with number of exhibits
- Consider adding formula: `state_dim = 2×n_exhibits + 5 + 128`

**Priority**: LOW - Paper is technically correct, but could be more explicit about scaling.

---

### 1.4 Conclude Subaction: SummarizeKeyPoints

**Issue**: Code has additional subaction `SummarizeKeyPoints` for Conclude option, but paper only mentions `WrapUp`

**Location in Code**:
- `src/utils/dialogue_planner.py:54-55`: Handles `SummarizeKeyPoints` subaction

**Location in Paper**:
- Line 459-461: Only mentions `WrapUp` for Conclude option

**Recommendation**: 
- Either add `SummarizeKeyPoints` to paper (if it's used)
- OR remove from code if it's not implemented/used
- Check if action masking prevents this from being selected

**Priority**: MEDIUM - Need to verify if this is actually used in practice.

---

### 1.5 OfferTransition Subaction Count

**Issue**: Paper suggests `OfferTransition` has only one subaction (`AssessAndTransition`), which matches code (`SuggestMove`), but the table description is more detailed than the code implementation

**Status**: This is actually correct - both paper and code have one subaction for OfferTransition. The discrepancy is just the naming (see 1.1).

**Priority**: N/A

---

### 1.6 Reward Weights and Parameters

**Issue**: Paper mentions weights `w_e` and `w_n` in base reward (line 498), but code comments indicate no weights for engagement/novelty

**Location in Paper**:
- Line 498: `R_t = w_e · r^{eng}_t + w_n · r^{nov}_t`
- Line 637: `r^{eng}_t = dwell_t` and `r^{nov}_t = 0.15 × |new facts|` (no weights mentioned)

**Location in Code**:
- `src/environment/env.py:63-69`: `w_engagement` defaults to 1.0, `novelty_per_fact` is 0.15 (scale factor, not weight)
- Line 280: `engagement_reward = max(0.0, self.dwell) * self.w_engagement`
- Line 284: `novelty_reward = len(new_fact_ids) * self.novelty_per_fact`

**Recommendation**: 
- Clarify in paper that base rewards are unweighted (w_e=1.0, w_n=1.0, with α=0.15 as scale factor)
- The extended reward function (line 634) correctly shows no weights for base components

**Priority**: LOW - Minor clarification needed.

---

### 1.7 Deliberation Cost

**Issue**: Code has `deliberation_cost` parameter but it's not mentioned in paper

**Location in Code**:
- `src/environment/env.py:52, 56`: `deliberation_cost=0.01` parameter exists but not used in reward calculation visible in `step()` method

**Recommendation**: 
- Either remove from code if unused, or add to paper if it's part of the reward function
- Check if this is legacy code

**Priority**: LOW - Need to verify if it's actually used.

---

## Part 2: Structural Issues and Recommendations

### 2.1 Duplicate Subsections

**Issue**: Two subsections with same title "Dialogue action tokens, fluent realization, and hierarchical control"

**Location**:
- Line 703-720: Under `\subsection{Simulator Design}`
- Line 725-737: Under `\subsection{Action Execution and Grounding}`

**Problem**: This creates confusion and redundancy. The content overlaps but is presented in different contexts.

**Recommendation**: 
- **Option A**: Merge into one subsection under "Action Execution and Grounding"
- **Option B**: Rename first to "Simulator Response Generation" and keep second as "Action Execution"
- **Option C**: Keep first as "Simulator Utterance Generation" and move "Dialogue action tokens" content to second subsection only

**Recommendation**: **Option B** - The first occurrence (line 703) is about simulator response generation, so rename it. The second (line 725) is about agent action execution, which is where dialogue action tokens belong.

---

### 2.2 Methodology Section Organization

**Issue**: The Methodology section mixes training algorithm, reward extensions, simulator design, action execution, and action masking in a way that doesn't follow a clear logical flow.

**Current Structure**:
1. Training Algorithm (line 524)
2. Reward Function Extensions (line 538)
3. Simulator Design (line 639)
4. Action Execution and Grounding (line 722)
5. Action Masking (line 818)

**Problems**:
- Simulator should come before/alongside training (it's the environment)
- Action execution and grounding should come before reward design (it's part of the base system)
- Reward extensions are correctly placed after base reward, but simulator design is too late

**Recommended Structure for Master's Thesis**:

```
Section 4: Problem Formulation (keep as is - this is good)
  - Hierarchical Framing
  - State Representation
  - Action Space
  - Reward Function Design (base rewards only)
  - Learning Objective
  - Summary

Section 5: System Architecture and Implementation
  - Overview
  - Environment Design
    - State Computation (DialogueBERT integration)
    - Action Space Implementation
    - Reward Calculation
  - Simulator Design
    - Architecture and Response Generation
    - Gaze Feature Synthesis
    - Transition Logic
    - Utterance Generation Modes
  - Agent Architecture
    - Actor-Critic Network
    - Option Selection and Termination
    - Action Masking
  - Action Execution and Grounding
    - Prompt Construction
    - Subaction Contracts
    - Fact Verification and Hallucination Detection

Section 6: Training Methodology
  - Training Algorithm (Actor-Critic for SMDPs)
  - Reward Function Extensions
    - Responsiveness Reward
    - Transition Control
    - Conclude Bonus
    - Combined Reward Function
  - Training Configuration
  - Evaluation Metrics
```

**Alternative (Keeping current section names but reorganizing)**:

```
Section 4: Problem Formulation (as is)

Section 5: System Architecture
  - Environment Implementation
  - Simulator Design
  - Agent Architecture
  - Action Execution and Grounding
  - Action Masking

Section 6: Training Methodology
  - Training Algorithm
  - Reward Function Extensions
  - Training Configuration
```

**Priority**: HIGH - This significantly improves thesis structure.

---

### 2.3 Simulator Placement

**Issue**: Simulator is described in Methodology section, but it's really part of the system architecture/environment

**Current Location**: Section 5 (Methodology), Subsection 5.3 (line 639)

**Problem**: Simulator is the environment - it should be described alongside the environment, not as a "training methodology" component.

**Recommendation**: Move simulator to System Architecture section, before Training Methodology.

**Priority**: HIGH - This is a fundamental organizational issue.

---

### 2.4 Formalization vs Implementation Separation

**Issue**: Problem Formulation section is good, but then Methodology jumps straight into implementation details without a clear "System Architecture" bridge.

**Current Flow**:
1. Problem Formulation (abstract, mathematical)
2. Methodology (jumps to training details)

**Recommended Flow**:
1. Problem Formulation (abstract, mathematical) ✓
2. **System Architecture** (how the abstract is realized)
   - Environment
   - Simulator
   - Agent
   - Action Execution
3. Training Methodology (how we train the system)
4. Evaluation

**Priority**: HIGH - This is standard thesis structure.

---

### 2.5 Reward Function Organization

**Issue**: Base reward is in Problem Formulation, but extensions are in Methodology. This splits the reward function description.

**Current**:
- Problem Formulation (line 475-502): Base rewards only
- Methodology (line 538-637): Reward extensions

**Problem**: Reader has to jump between sections to understand full reward function.

**Recommendation**:
- **Option A**: Keep base rewards in Problem Formulation, but add a brief summary of extensions there too
- **Option B**: Move all reward discussion to Methodology (but this loses the clean formalization)
- **Option C**: Keep as is but add cross-references and a summary table

**Recommendation**: **Option C** - Add a summary equation in Problem Formulation that references Methodology for details, and add a clear summary table in Methodology showing all components.

**Priority**: MEDIUM - Current structure is acceptable with better cross-references.

---

### 2.6 Action Execution Content Placement

**Issue**: Some content about action execution appears in Simulator section (line 740 - template/LLM modes), but it's actually about agent response generation.

**Location**: Line 740 mentions "Template mode" and "LLM mode" in simulator section, but this is about agent utterance generation, not simulator responses.

**Recommendation**: 
- Move this content to "Action Execution and Grounding" section
- Keep simulator's utterance generation separate (it's about visitor responses)

**Priority**: MEDIUM - Content is misplaced.

---

## Part 3: Specific Edit Recommendations

### 3.1 Fix Subaction Naming

**File**: `paper.tex`

**Change 1**: Line 455 (Table)
```latex
\textsc{OfferTransition} & \textit{SuggestMove}
```

**Change 2**: Line 472
```latex
\textit{SuggestMove} \emph{computes coverage across exhibits...
```

**Change 3**: Line 777
```latex
\texttt{OPTION: OfferTransition | SUBACTION: SuggestMove | TARGET: [e*]}.
```

---

### 3.2 Clarify Transition Spam Implementation

**File**: `paper.tex`

**Location**: Line 601-606, 634

**Change**: Update to clarify simulator-level handling:

```latex
\paragraph{Transition spam control:} To prevent \textit{transition spam}—repeated \textsc{OfferTransition} attempts without intervening content—we implement a simulator-level penalty that reduces dwell time for consecutive transitions. The simulator tracks consecutive transition attempts and applies a multiplier to dwell values (reducing $r^{\text{eng}}_t$), with penalties escalating after the first two consecutive transitions. This mechanism is implemented at the simulator level rather than as an explicit reward component, ensuring that transition spam naturally reduces engagement signals rather than requiring separate penalty terms.
```

**Also update line 634**: Remove `r^{spam}_t` from equation or note it's implicit:
```latex
R_t = r^{\text{eng}}_t + r^{\text{nov}}_t + r^{\text{resp}}_t + r^{\text{trans}}_t + r^{\text{conclude}}_t
```

(Note: Transition spam affects $r^{\text{eng}}_t$ via simulator-level dwell reduction)

---

### 3.3 Remove Duplicate Subsection

**File**: `paper.tex`

**Location**: Line 703-720

**Action**: Rename subsection to:
```latex
\subsubsection{Simulator Utterance Generation: Template and LLM Modes}
```

Move dialogue action tokens discussion to Action Execution section only.

---

### 3.4 Add State Dimension Formula

**File**: `paper.tex`

**Location**: After line 412

**Add**:
```latex
\noindent\textbf{State dimension scaling:} For $n$ exhibits, the total dimension is $2n + 5 + 128 = 2n + 133$. The formula breaks down as: focus vector ($n+1$), history vector ($n+4$), intent embedding ($64$), and dialogue context ($64$). This design scales linearly with the number of exhibits while keeping the semantic embeddings fixed at 128 dimensions total.
```

---

### 3.5 Reorganize Sections (Major Restructure)

**Recommended Section Structure**:

```latex
\section{Introduction} % (as is)

\section{Background and Related Works} % (as is)

\section{Research Questions} % (as is)

\section{Problem Formulation} % (as is - keep mathematical formalization)

\section{System Architecture} % NEW SECTION
\subsection{Environment Implementation}
\subsubsection{State Space Computation}
\subsubsection{Action Space and Masking}
\subsubsection{Reward Calculation}
\subsection{Simulator Design}
\subsubsection{Architecture and Response Generation}
\subsubsection{Gaze Feature Synthesis}
\subsubsection{Transition Logic}
\subsubsection{Utterance Generation Modes}
\subsection{Agent Architecture}
\subsubsection{Actor-Critic Network}
\subsubsection{Option Selection and Termination}
\subsection{Action Execution and Grounding}
\subsubsection{Prompt Construction and Subaction Contracts}
\subsubsection{Fact Verification and Hallucination Detection}

\section{Training Methodology} % RENAMED FROM "Methodology"
\subsection{Training Algorithm}
\subsection{Reward Function Extensions}
\subsubsection{Responsiveness Reward}
\subsubsection{Transition Control}
\subsubsection{Conclude Bonus}
\subsubsection{Combined Reward Function}
\subsection{Training Configuration}

\section{Hypotheses} % (as is)

\section{Evaluation/Experimental Design} % (as is)

\section{Results} % (as is)

\section{Discussion} % (as is)
```

---

## Part 4: Summary of Priority Actions

### HIGH PRIORITY (Fix Immediately)
1. ✅ Fix subaction naming: `AssessAndTransition` → `SuggestMove` (3 locations)
2. ✅ Reorganize sections: Create "System Architecture" section, move simulator there
3. ✅ Remove duplicate subsection: Rename first "Dialogue action tokens" subsection
4. ✅ Clarify transition spam: Update to reflect simulator-level implementation

### MEDIUM PRIORITY (Improve Clarity)
5. ⚠️ Add state dimension scaling formula
6. ⚠️ Move misplaced content (template/LLM modes) to correct section
7. ⚠️ Add cross-references between Problem Formulation and Methodology for rewards
8. ⚠️ Verify and document `SummarizeKeyPoints` subaction (use or remove)

### LOW PRIORITY (Nice to Have)
9. 📝 Clarify reward weights (w_e, w_n) in base reward section
10. 📝 Check if `deliberation_cost` is used, document or remove
11. 📝 Add more explicit formula for state dimension scaling

---

## Part 5: Additional Observations

### Positive Aspects
- Problem Formulation section is well-structured and mathematically rigorous
- Reward function extensions are well-motivated
- Simulator design is detailed and well-explained
- Action masking is clearly described

### Areas for Improvement
- Better separation between "what" (formalization) and "how" (implementation)
- Clearer flow from abstract to concrete
- Better cross-referencing between sections
- More explicit connection between design decisions and implementation

---

## Next Steps

1. Review this document
2. Prioritize which changes to implement
3. Create updated `paper.tex` with recommended changes
4. Verify all codebase discrepancies are resolved
5. Ensure thesis structure follows standard format for master's thesis

