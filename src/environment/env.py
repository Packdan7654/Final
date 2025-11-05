"""
Hierarchical Reinforcement Learning Environment for Museum Dialogue Agent

This module implements a Semi-Markov Decision Process (SMDP) environment for museum
dialogue agents using the Options Framework (Sutton et al., 1999). The environment
supports temporal abstraction through high-level options and low-level subactions,
enabling efficient learning in sparse-reward, long-horizon dialogue settings.

Key HRL Components:
- Options Framework: High-level strategies (Explain, Ask, Transition, Conclude) per paper.tex
- Intra-option policies: Low-level subactions within each option
- Learned termination: Option-Critic style termination functions
- Action masking: Prevents invalid actions based on dialogue state
- Gaze-based rewards: Engagement signals from dwell time (Bozkir et al., 2021)
- Dynamic Knowledge Graph: Flexible loading from Neo4j or fallback sources

References:
- Sutton, R. S., Precup, D., & Singh, S. (1999). Between MDPs and semi-MDPs: A framework 
  for temporal abstraction in reinforcement learning. Artificial Intelligence.
- Bacon, P. L., Harb, J., & Precup, D. (2017). The option-critic architecture. AAAI.
- Bozkir, E., et al. (2021). Eye tracking in virtual learning environments.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import re
import os
from typing import Dict, Set
from src.utils.dialogue_planner import build_prompt 
from src.utils.knowledge_graph import SimpleKnowledgeGraph
from src.utils.dialoguebert_intent_recognizer import get_dialoguebert_recognizer
from typing import Dict, List, Optional, Any


class MuseumDialogueEnv(gym.Env):
    """
    Hierarchical Reinforcement Learning Environment for Museum Dialogue Agent
    
    Implements the Options Framework as a Semi-Markov Decision Process (SMDP):
    - High-level options (temporally extended actions) select dialogue strategies
    - Low-level subactions specify concrete utterance types within each option
    - Learned termination functions determine when to switch options
    - Action masking prevents invalid actions based on dialogue state
    
    State Space: Focus snapshot + dialogue history + intent label + trend features
    Action Space: Hierarchical (option, subaction, terminate_option)
    Reward: Engagement (dwell) + Novelty (coverage) + deliberation cost
    """
    
    def __init__(self, deliberation_cost=0.01, knowledge_graph_path=None, max_turns=20):
        super().__init__()
        
        # ===== CONFIGURATION =====
        self.deliberation_cost = deliberation_cost
        self.max_turns = max_turns  # Maximum turns per episode
        
        # ===== REWARD PARAMETERS (configurable via environment variables) =====
        # Default values match paper baseline (per paper.tex Section 4.7, line 637)
        # Note: Per paper, r^eng_t = dwell_t and r^nov_t = 0.15 × |new facts| (no weights)
        
        # Engagement: r^eng_t = dwell_t (no weight, just dwell time)
        # Note: We keep w_engagement for backward compatibility but it should be 1.0
        self.w_engagement = float(os.environ.get("HRL_W_ENGAGEMENT", "1.0"))
        
        # Novelty: r^nov_t = α × |new facts| where α = 0.15 (per paper.tex line 637)
        # This is the scale factor α, not a separate weight
        self.novelty_per_fact = float(os.environ.get("HRL_NOVELTY_PER_FACT", "0.15"))
        
        # Responsiveness reward scale (per paper.tex Section 4.7)
        self.w_responsiveness = float(os.environ.get("HRL_W_RESPONSIVENESS", "0.25"))
        
        # Conclude bonus scale (per paper.tex Section 4.7)
        self.w_conclude = float(os.environ.get("HRL_W_CONCLUDE", "0.2"))
        
        # Transition insufficiency penalty parameter (per paper.tex Section 4.7)
        # Note: Transition spam is handled at simulator level (reduces dwell time)
        self.w_transition_insufficiency = float(os.environ.get("HRL_W_TRANSITION_INSUFFICIENCY", "-0.20"))
        
        # ===== HIERARCHICAL ACTION SPACE =====
        # High-level options represent dialogue strategies (per paper.tex)
        self.options = ["Explain", "AskQuestion", "OfferTransition", "Conclude"]
        
        # Low-level subactions within each option (per paper.tex Table 1)
        self.subactions = {
            "Explain": ["ExplainNewFact", "RepeatFact", "ClarifyFact"],
            "AskQuestion": ["AskOpinion", "AskMemory", "AskClarification"],
            "OfferTransition": ["SuggestMove"],
            "Conclude": ["WrapUp"]
        }
        
        # Action masking parameters for dialogue coherence
        self.min_facts_before_conclude = 3
        self.min_exhibits_before_conclude = 2
        
        # ===== SIMPLIFIED KNOWLEDGE GRAPH =====
        # Load knowledge graph from JSON or use default
        self.knowledge_graph = SimpleKnowledgeGraph(knowledge_graph_path)
        
        # Extract knowledge graph data
        self.exhibit_keys = self.knowledge_graph.get_exhibit_names()
        self.n_exhibits = len(self.exhibit_keys)
        
        # ===== OBSERVATION SPACE =====
        # State representation with DialogueBERT as per paper formalization (Section 4.6):
        # s_t = [f_t, h_t, i_t, c_t]
        # - f_t: focus vector (n_exhibits + 1 for no-focus)
        # - h_t: dialogue history (n_exhibits completion ratios + 4 option usage)
        # - i_t: intent embedding (64-d projection from 768-d DialogueBERT)
        # - c_t: dialogue context (64-d projection from 768-d DialogueBERT)
        # Total dimensions calculated dynamically based on number of exhibits
        # Example: 5 exhibits → (5+1) + (5+4) + 64 + 64 = 143-d
        
        focus_dim = self.n_exhibits + 1  # +1 for "no focus" state
        history_dim = self.n_exhibits + len(self.options)  # completion ratios + actions used
        intent_dim = 64  # Projected DialogueBERT intent embedding
        context_dim = 64  # Projected DialogueBERT dialogue context

        total_obs_dim = focus_dim + history_dim + intent_dim + context_dim
        
        print(f"[Environment] Observation space: {total_obs_dim}-d "
              f"(focus={focus_dim}, history={history_dim}, intent={intent_dim}, context={context_dim})")
        
        self.observation_space = spaces.Box(
            low=-10.0,  # Allow negative values after projection
            high=10.0, 
            shape=(total_obs_dim,), 
            dtype=np.float32
        )
        
        # ===== DIALOGUEBERT PROJECTION MATRIX =====
        # Fixed, offline-trained linear projection P: R^768 -> R^64
        # This implements the projection: i_t = P * e_t where e_t is the 768-d DialogueBERT output
        # Using a fixed random projection with normalization (Johnson-Lindenstrauss type)
        np.random.seed(42)  # Fixed seed for reproducibility
        self.projection_matrix = np.random.randn(64, 768).astype(np.float32) / np.sqrt(768)
        
        # ===== HIERARCHICAL ACTION SPACE =====
        # SMDP action space with options, subactions, and termination
        self.action_space = spaces.Dict({
            "option": spaces.Discrete(len(self.options)),
            "subaction": spaces.Discrete(max(len(subacts) for subacts in self.subactions.values())),
            "terminate_option": spaces.Discrete(2)
        })

        # Initialize environment state
        self.reset()

    # ===== ENVIRONMENT LIFECYCLE =====
    
    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        # Core dialogue state
        self.focus = 0  # Current exhibit focus (0 = none)
        self.dwell = 0.0  # Current dwell time (engagement signal)
        self.last_user_utterance = ""
        self._previous_dwell = 0.0  # Store previous dwell for lagged rewards
        self._last_user_intent = "statement"  # Track user intent for responsiveness
        self._last_user_response_type = "statement"  # Track simulator's response type (question, confusion, etc.)
        
        # Dialogue history for context
        self.dialogue_history = []  # List of (role, utterance) tuples
        self.max_dialogue_history = 6  # Keep last 6 utterances (3 exchanges)
        
        # SIMPLE FACT TRACKING: exhibit_name -> set of fact IDs mentioned
        self.facts_mentioned_per_exhibit: Dict[str, Set[str]] = {ex: set() for ex in self.exhibit_keys}
        self.facts_mentioned_this_turn: Set[str] = set()  # Fact IDs mentioned this turn only
        
        # Dialogue history tracking
        self.explained = [0] * self.n_exhibits  # Which exhibits have been explained
        self.actions_used = {opt: 0 for opt in self.options}  # Count of each action type used
        
        # Option tracking (for termination learning)
        self.current_option = "Explain"
        self.turns_in_option = 0
        self._previous_option = None
        
        # Transition insufficiency tracking (per paper.tex Section 4.7)
        # Track successful transitions and turns since for 3-turn exemption rule
        self.successful_transition_turns = []  # List of turn numbers when successful transitions occurred
        
        # Note: Trend tracking removed to match paper specification
        
        # Session tracking
        self.turn_count = 0
        self.session_reward = 0.0
        
        # DialogueBERT embeddings for insight reporting (stored as full 768-d for visualization)
        # Note: The state uses 64-d projected versions, but we keep 768-d for diagnostics
        self._last_intent_embedding = np.zeros(768, dtype=np.float32)
        self._last_dialogue_context = np.zeros(768, dtype=np.float32)
        self._prev_intent_embedding = np.zeros(768, dtype=np.float32)
        self._prev_dialogue_context = np.zeros(768, dtype=np.float32)
        
        # Initialize action masks
        self.action_masks = [True] * len(self.options)  # All options available initially
        
        return self._get_obs(), {}

    def step(self, action_dict):
        """Execute one step in the SMDP environment"""
        # Apply action masking
        masked_action = self._apply_action_masks(action_dict)
        
        # Extract actions
        option_idx = masked_action["option"]
        subaction_idx = masked_action["subaction"]
        terminate_option = masked_action["terminate_option"]
        
        # Get available options and subactions
        available_options = self._get_available_options()
        option = available_options[option_idx]
        available_subactions = self._get_available_subactions(option)
        subaction = available_subactions[subaction_idx]
        
        # Handle option termination and transitions (Option-Critic style)
        if self.current_option != option or (terminate_option and self.current_option is not None):
            # Option is switching/terminating
            self.current_option = option
            self.option_start_turn = self.turn_count
            self.turns_in_option = 0
        else:
            # Continue current option
            self.turns_in_option += 1
        
        # Update action usage
        self.actions_used[option] += 1
        
        # Get current exhibit ID for response generation and reward calculation
        ex_id = self._get_current_exhibit()
        
        # Generate agent response using dialogue planner and LLM
        agent_response, target_exhibit = self._generate_agent_response(option, subaction)
        
        # Store agent utterance in dialogue history
        self._update_dialogue_history("agent", agent_response)
        
        # === SIMPLE FACT EXTRACTION WITH VALIDATION ===
        # Find [ID] patterns in agent response
        fact_ids_mentioned = re.findall(r'\[([A-Z]{2}_\d{3})\]', agent_response)
        
        # Check verbose mode
        import os
        verbose = os.environ.get('HRL_VERBOSE', '0') == '1'
        
        # Get all valid fact IDs for current exhibit
        valid_fact_ids = set()
        for fact_with_id in self.knowledge_graph.get_exhibit_facts(ex_id):
            fact_id = self.knowledge_graph.extract_fact_id(fact_with_id)
            if fact_id:
                valid_fact_ids.add(fact_id)
        
        # Track NEW and VALID facts only
        new_fact_ids = []
        hallucinated_ids = []
        for fact_id in fact_ids_mentioned:
            # Check if fact ID is valid for this exhibit
            if fact_id not in valid_fact_ids:
                hallucinated_ids.append(fact_id)
                if verbose:
                    print(f"HALLUCINATED: [{fact_id}] (not a real fact for {ex_id})")
                continue
            
            # Check if already mentioned
            if fact_id not in self.facts_mentioned_per_exhibit[ex_id]:
                new_fact_ids.append(fact_id)
                self.facts_mentioned_per_exhibit[ex_id].add(fact_id)
                self.facts_mentioned_this_turn.add(fact_id)
                if verbose:
                    print(f"NEW FACT: [{fact_id}]")
            else:
                if verbose:
                    print(f"REPEAT FACT: [{fact_id}]")
        
        # ===== REWARD CALCULATION (per paper formalization, Section 4.7, line 637) =====
        # Engagement reward: r^eng_t = dwell_t (per paper.tex line 637)
        engagement_reward = max(0.0, self.dwell) * self.w_engagement
        
        # Novelty reward: r^nov_t = 0.15 × |new facts at t| (per paper.tex line 637)
        # No separate weight - the 0.15 is the scale factor α itself
        novelty_reward = len(new_fact_ids) * self.novelty_per_fact
        
        # Responsiveness reward: Incentivize answering visitor questions (Grice's Maxim of Quantity)
        responsiveness_reward = 0.0
        if self._last_user_response_type == "question":
            # Visitor asked a question last turn
            if len(new_fact_ids) > 0:
                # Agent provided NEW facts this turn (answered the question) → POSITIVE REWARD
                responsiveness_reward = +self.w_responsiveness
                if verbose:
                    print(f"✅ RESPONSIVENESS: +{self.w_responsiveness:.2f} (answered visitor's question with {len(new_fact_ids)} fact(s))")
            elif option == "AskQuestion":
                # Agent deflected with counter-question instead of answering → NEGATIVE PUNISHMENT
                responsiveness_reward = -self.w_responsiveness * 0.6  # 60% of positive reward
                if verbose:
                    print(f"❌ RESPONSIVENESS: {responsiveness_reward:.2f} (deflected visitor's question with counter-question)")
        
        # Conclude bonus: scales with number of exhibits covered (weighted)
        conclude_bonus = 0.0
        if option == "Conclude":
            # Count exhibits with at least 1 fact mentioned
            exhibits_with_facts = sum(1 for fact_set in self.facts_mentioned_per_exhibit.values() 
                                     if len(fact_set) > 0)
            # Bonus scales with weight per exhibit covered
            conclude_bonus = exhibits_with_facts * self.w_conclude
            if verbose:
                print(f"🎯 CONCLUDE BONUS: +{conclude_bonus:.2f} ({exhibits_with_facts} exhibits covered)")
        
        # Transition insufficiency penalty (per paper.tex Section 4.7: Transition Control)
        # Note: Transition spam is handled at simulator level (reduces dwell time)
        transition_insufficiency_penalty = 0.0
        
        if option == "OfferTransition":
            current_exhibit = self._get_current_exhibit()
            facts_shared_at_current = len(self.facts_mentioned_per_exhibit[current_exhibit])
            
            # Check if we're within 3 turns of a successful transition (exemption rule)
            # Remove transitions older than 3 turns
            self.successful_transition_turns = [t for t in self.successful_transition_turns 
                                             if (self.turn_count + 1 - t) <= 3]
            
            # Check if we're exempt due to recent successful transition
            is_exempt = len(self.successful_transition_turns) > 0
            
            # Transition insufficiency penalty: scales with fewer facts
            # The fewer facts, the bigger the penalty
            # Base penalty at 0 facts, scales down as facts increase
            if facts_shared_at_current == 0:
                # Maximum penalty when no facts shared
                penalty_scale = 1.0
            elif facts_shared_at_current == 1:
                # Medium-high penalty for only 1 fact
                penalty_scale = 0.8
            else:
                # No penalty for 2+ facts
                penalty_scale = 0.0
            
            # Apply penalty only if:
            # 1. We have insufficient facts (penalty_scale > 0)
            # 2. We're NOT exempt due to successful transition in last 3 turns
            if penalty_scale > 0 and not is_exempt:
                transition_insufficiency_penalty = self.w_transition_insufficiency * penalty_scale
                if verbose:
                    print(f"⚠️  TRANSITION INSUFFICIENCY PENALTY: {transition_insufficiency_penalty:.2f} (only {facts_shared_at_current} fact(s) at {current_exhibit}, scale={penalty_scale:.1f})")
            elif is_exempt and verbose:
                print(f"✅ TRANSITION INSUFFICIENCY EXEMPT: Successful transition in last 3 turns (turns since: {[self.turn_count + 1 - t for t in self.successful_transition_turns]})")
        
        # Clear this turn's tracking for next turn
        facts_shared_count_this_turn = len(new_fact_ids)
        self.facts_mentioned_this_turn = set()
        
        # Total reward: R_t = r^eng + r^nov + r^resp + r^trans + r^conclude (per paper.tex line 634)
        # Note: r^eng_t = dwell_t, r^nov_t = 0.15 × |new facts|, no weights per paper
        # Transition insufficiency is explicit reward component (per paper.tex Section 4.7)
        # Transition spam is handled at simulator level (reduces dwell time)
        step_reward = engagement_reward + novelty_reward + responsiveness_reward + conclude_bonus + transition_insufficiency_penalty
        
        # Store current dwell for NEXT turn's reward
        self._previous_dwell = self.dwell
        
        # Update session reward
        self.session_reward += step_reward
        
        # Update exhibits covered count
        exhibits_covered = sum(1 for exp in self.explained if exp > 0)
        
        # Update turn count
        self.turn_count += 1
        
        # Check termination conditions
        done = (option == "Conclude" or self.turn_count >= self.max_turns)
        
        # Calculate current exhibit completion rate for transition logic
        current_exhibit_completion = len(self.facts_mentioned_per_exhibit[ex_id]) / len(self.knowledge_graph.get_exhibit_facts(ex_id)) if len(self.knowledge_graph.get_exhibit_facts(ex_id)) > 0 else 0.0
        
        # Build info dictionary
        info = {
            "agent_utterance": agent_response,
            "option": option,
            "subaction": subaction,
            "terminated_option": terminate_option,
            "reward_engagement": engagement_reward,
            "reward_novelty": novelty_reward,
            "reward_responsiveness": responsiveness_reward,
            "reward_conclude": conclude_bonus,
            "reward_transition_insufficiency": transition_insufficiency_penalty,
            "dwell": self.dwell,
            "total_reward": step_reward,
            "session_reward": self.session_reward,
            "turn": self.turn_count,
            "facts_shared": facts_shared_count_this_turn,
            "facts_mentioned_in_utterance": new_fact_ids,
            "hallucinated_facts": hallucinated_ids,
            "exhibits_covered": exhibits_covered,
            "current_option": self.current_option,
            "turns_in_option": self.turns_in_option,
            "current_focus": self.focus,
            "current_exhibit": ex_id,
            "target_exhibit": target_exhibit,  # For OfferTransition: where agent wants to go
            "current_exhibit_completion": current_exhibit_completion,  # For transition probability
            "available_options": self._get_available_options(),
            "available_subactions": self._get_available_subactions(option),
            "action_masks": self.get_action_masks(),
            "agent_llm_time": getattr(self, '_last_agent_llm_time', 0.0)
        }
        
        # ===== DialogueBERT INSIGHTS (for visualization) =====
        # Note: DialogueBERT insights will be added after user state is updated
        
        return self._get_obs(), step_reward, done, False, info

    # ===== OBSERVATION CONSTRUCTION =====
    
    def _get_obs(self):
        """
        Construct observation vector with DialogueBERT as per paper formalization (Section 4.6):
        s_t = [f_t, h_t, i_t, c_t]
        
        Where:
        - f_t: focus vector (one-hot over exhibits + no-focus)
        - h_t: dialogue history (exhibits explained + action counts)
        - i_t: projected intent embedding = P * DialogueBERT(u_t, role="user")
        - c_t: projected dialogue context = P * (1/3) * Σ DialogueBERT(recent_utterances)
        
        Projection: 768-d -> 64-d using fixed matrix P
        """
        
        # 1. Focus vector f_t (9-d)
        focus_snapshot = np.zeros(self.n_exhibits + 1)
        if self.focus > 0:
            focus_snapshot[self.focus - 1] = 1.0
        else:
            focus_snapshot[-1] = 1.0  # No focus
        
        # 2. Dialogue history vector h_t (12-d) - Updated per paper spec
        # First 8: exhibit completion ratios (0-1 for facts shared per exhibit)
        # Next 4: option usage (normalized counts)
        history = np.zeros(self.n_exhibits + len(self.options))

        # Get current exhibit completion data
        coverage = self._get_museum_exhibit_coverage()

        # Exhibit completion ratios (0-1 for facts shared) - positions 0-7
        for i, exhibit_name in enumerate(self.exhibit_keys):
            completion_ratio = coverage.get(exhibit_name, {"coverage": 0.0})["coverage"]
            history[i] = completion_ratio

        # Actions used (normalized counts) - positions 8-11
        total_actions = sum(self.actions_used.values()) or 1
        for i, opt in enumerate(self.options):
            history[self.n_exhibits + i] = self.actions_used[opt] / total_actions
        
        # 3. Intent embedding i_t (64-d projected from 768-d)
        # Get DialogueBERT embedding: e_t = DialogueBERT(u_t, role="user")
        intent_recognizer = get_dialoguebert_recognizer()
        intent_embedding_768 = intent_recognizer.get_intent_embedding(
            self.last_user_utterance, role="user"
        )
        
        # Apply projection: i_t = P * e_t
        intent_embedding_64 = np.dot(self.projection_matrix, intent_embedding_768).astype(np.float32)
        
        # 4. Dialogue context c_t (64-d projected from 768-d)
        # Get DialogueBERT context embedding (average of last 3 turns)
        # dialogue_history already contains (role, utterance) tuples
        dialogue_context_768 = intent_recognizer.get_dialogue_context(
            self.dialogue_history, max_turns=3
        )
        
        # Apply projection: c_t = P * context_768
        dialogue_context_64 = np.dot(self.projection_matrix, dialogue_context_768).astype(np.float32)
        
        # Track embeddings for insights (keep full 768-d for visualization)
        prev_intent = getattr(self, '_last_intent_embedding', np.zeros(768, dtype=np.float32))
        prev_context = getattr(self, '_last_dialogue_context', np.zeros(768, dtype=np.float32))
        self._prev_intent_embedding = prev_intent.astype(np.float32)
        self._prev_dialogue_context = prev_context.astype(np.float32)
        self._last_intent_embedding = intent_embedding_768.astype(np.float32)
        self._last_dialogue_context = dialogue_context_768.astype(np.float32)
        
        # Concatenate into observation vector: [f_t, h_t, i_t, c_t]
        # Total: 9 + 12 + 64 + 64 = 149-d
        obs = np.concatenate([
            focus_snapshot,        # 9-d
            history,               # 12-d
            intent_embedding_64,   # 64-d
            dialogue_context_64    # 64-d
        ]).astype(np.float32)
        
        return obs

    # ===== MUSEUM OVERVIEW =====
    
    def _get_museum_exhibit_coverage(self):
        """
        Calculate coverage stats for all exhibits in the museum.
        Returns dict: {exhibit_name: {"total": int, "mentioned": int, "coverage": float}}
        """
        coverage = {}
        for exhibit_name in self.exhibit_keys:
            all_facts = self.knowledge_graph.get_exhibit_facts(exhibit_name)
            total_facts = len(all_facts)
            mentioned_facts = len(self.facts_mentioned_per_exhibit[exhibit_name])
            coverage[exhibit_name] = {
                "total": total_facts,
                "mentioned": mentioned_facts,
                "coverage": mentioned_facts / total_facts if total_facts > 0 else 0.0
            }
        return coverage
    
    def _select_least_discussed_exhibit(self, current_exhibit: str) -> str:
        """
        Select the exhibit with the least coverage (excluding current exhibit).
        If multiple exhibits tie for least coverage, choose randomly among them.
        """
        coverage = self._get_museum_exhibit_coverage()
        
        # Remove current exhibit from consideration
        other_exhibits = {k: v for k, v in coverage.items() if k != current_exhibit}
        
        if not other_exhibits:
            # Fallback: stay at current exhibit (shouldn't happen)
            return current_exhibit
        
        # Find minimum coverage percentage
        min_coverage = min(v["coverage"] for v in other_exhibits.values())
        
        # Get all exhibits with minimum coverage
        least_discussed = [k for k, v in other_exhibits.items() if v["coverage"] == min_coverage]
        
        # Random choice if multiple exhibits tie
        return self.np_random.choice(least_discussed)
    
    # ===== ACTION MASKING =====
    
    def _get_available_options(self):
        """Get available options based on current state (action masking)"""
        available_options = self.options.copy()
        
        # Count total facts mentioned across all exhibits
        total_facts_mentioned = sum(len(ids) for ids in self.facts_mentioned_per_exhibit.values())
        
        # Check if enough facts have been shared to allow conclusion
        if total_facts_mentioned < self.min_facts_before_conclude:
            if "Conclude" in available_options:
                available_options.remove("Conclude")
        
        # Check if enough exhibits have been covered
        exhibits_covered = sum(1 for ids in self.facts_mentioned_per_exhibit.values() if len(ids) > 0)
        if exhibits_covered < self.min_exhibits_before_conclude:
            if "Conclude" in available_options:
                available_options.remove("Conclude")
        
        return available_options

    def _get_available_subactions(self, option):
        """Get available subactions for a given option (action masking)"""
        if option not in self.subactions:
            return []
        
        subactions = self.subactions[option].copy()
        
        # Only apply basic fact availability masking for Explain option
        if option == "Explain":
            current_exhibit = self.exhibit_keys[self.focus - 1] if self.focus > 0 else None
            if current_exhibit:
                all_facts = self.knowledge_graph.get_exhibit_facts(current_exhibit)
                mentioned_ids = self.facts_mentioned_per_exhibit[current_exhibit]
                
                # Check if there are unmentioned facts
                has_unmentioned = any(self.knowledge_graph.extract_fact_id(f) not in mentioned_ids 
                                     for f in all_facts)
                
                # Remove ExplainNewFact if all facts mentioned
                if not has_unmentioned and "ExplainNewFact" in subactions:
                    subactions.remove("ExplainNewFact")
                
                # Remove RepeatFact if no facts mentioned yet
                if len(mentioned_ids) == 0 and "RepeatFact" in subactions:
                    subactions.remove("RepeatFact")
        
        return subactions

    def _apply_action_masks(self, action_dict):
        """Apply action masks to ensure valid actions"""
        masked_action = action_dict.copy()
        
        # Mask options
        available_options = self._get_available_options()
        if masked_action["option"] >= len(available_options):
            masked_action["option"] = 0  # Default to first available
        
        # Mask subactions
        option = available_options[masked_action["option"]]
        available_subactions = self._get_available_subactions(option)
        if masked_action["subaction"] >= len(available_subactions):
            masked_action["subaction"] = 0  # Default to first available
        
        return masked_action

    def get_action_masks(self):
        """Get current action masks for training"""
        available_options = self._get_available_options()
        option_mask = [1 if opt in available_options else 0 for opt in self.options]
        
        # Get subaction mask for first available option
        if available_options:
            first_option = available_options[0]
            available_subactions = self._get_available_subactions(first_option)
            subaction_mask = [1 if sub in available_subactions else 0 for sub in self.subactions[first_option]]
        else:
            subaction_mask = [0] * max(len(subacts) for subacts in self.subactions.values())
        
        return {
            "option_mask": option_mask,
            "subaction_mask": subaction_mask
        }

    # ===== AGENT RESPONSE GENERATION =====
    
    def _generate_agent_response(self, option: str, subaction: str) -> tuple:
        """Generate agent response using dialogue planner and LLM
        
        Returns:
            tuple: (response_str, target_exhibit_str or None)
        """
        try:
            # Get current exhibit ID
            ex_id = self._get_current_exhibit()
            all_facts = self.knowledge_graph.get_exhibit_facts(ex_id)
            
            # SIMPLE: Get unmentioned facts for this exhibit
            mentioned_ids = self.facts_mentioned_per_exhibit[ex_id]
            unmentioned_facts = [f for f in all_facts 
                               if self.knowledge_graph.extract_fact_id(f) not in mentioned_ids]
            
            # FOR TRANSITION: Calculate target exhibit and museum coverage
            target_exhibit = None
            coverage_dict = None
            if option == "OfferTransition":
                target_exhibit = self._select_least_discussed_exhibit(ex_id)
                coverage_dict = self._get_museum_exhibit_coverage()
            
            # Get facts based on action type
            if option == "Explain":
                # For Explain actions, show unmentioned facts
                facts_for_prompt = unmentioned_facts
            else:
                # For other actions, show all facts (mentioned + unmentioned)
                facts_for_prompt = all_facts

            # Get mentioned facts for RepeatFact actions
            mentioned_facts = [f for f in all_facts
                              if self.knowledge_graph.extract_fact_id(f) in self.facts_mentioned_per_exhibit[ex_id]]

            # Build prompt with dialogue history
            prompt = build_prompt(
                option=option,
                subaction=subaction,
                ex_id=ex_id,
                last_utt=self.last_user_utterance,
                facts_all=facts_for_prompt,
                facts_used=mentioned_facts,  # Pass mentioned facts for RepeatFact
                selected_fact=None,
                dialogue_history=self.dialogue_history,
                exhibit_names=self.exhibit_keys,
                knowledge_graph=self.knowledge_graph,
                target_exhibit=target_exhibit,  # NEW: For transitions
                coverage_dict=coverage_dict    # NEW: For transitions
            )
            
            # Generate response using LLM from centralized config
            import time
            import os
            verbose = os.environ.get('HRL_VERBOSE', '0') == '1'
            start_time = time.time()
            if verbose:
                print(f"[Agent LLM] Generating response for {option}/{subaction}...", flush=True)
                print(f"[Agent LLM] PROMPT:\n{'-'*60}\n{prompt}\n{'-'*60}", flush=True)
            from LLM_CONFIG import get_agent_llm
            llm = get_agent_llm()
            
            system_prompt = """You are a museum guide. Be natural and conversational.
When explaining facts, you MUST include the fact ID in brackets.
Example: "Created by Gerrit Dou in 1635 [TU_003]"
Keep responses 2-3 sentences."""
            
            response = llm.generate(prompt, system_prompt=system_prompt)
            elapsed = time.time() - start_time
            if verbose:
                print(f"[Agent LLM] Response received in {elapsed:.2f}s ({len(response)} chars)", flush=True)
            
            # Store prompt and timing for debugging and logging
            self._last_llm_prompt = prompt
            self._last_agent_system_prompt = system_prompt
            self._last_agent_llm_time = elapsed
            
            return response.strip(), target_exhibit
            
        except Exception as e:
            print(f"Error generating agent response: {e}")
            raise e



    # ===== HELPER METHODS =====
    
    def _get_current_exhibit(self) -> str:
        """Get current exhibit ID based on focus"""
        if self.focus > 0 and self.focus <= len(self.exhibit_keys):
            return self.exhibit_keys[self.focus - 1]
        # Fallback to first exhibit if focus is invalid (should not happen in normal operation)
        if len(self.exhibit_keys) > 0:
            print(f"⚠️  WARNING: Invalid focus {self.focus}, falling back to {self.exhibit_keys[0]}")
            return self.exhibit_keys[0]
        return "Unknown"  # Should never happen if knowledge graph is loaded


    def record_successful_transition(self):
        """Record that a transition was successful (called from training loop after simulator responds).
        
        This is used for the 3-turn exemption rule: if a transition succeeds,
        the insufficiency penalty is not applied for the next 3 turns.
        """
        self.successful_transition_turns.append(self.turn_count)
    
    def _update_dialogue_history(self, role: str, utterance: str):
        """Update dialogue history with new utterance
        
        Args:
            role: 'agent' or 'user'
            utterance: The text utterance
        """
        if utterance and utterance.strip():
            self.dialogue_history.append((role, utterance))
            # Keep only recent utterances
            if len(self.dialogue_history) > self.max_dialogue_history:
                self.dialogue_history = self.dialogue_history[-self.max_dialogue_history:]

    def update_user_state(self, focus: int = None, dwell: float = None, utterance: str = None, 
                          response_type: str = None, simulator=None):
        """Update user state from simulator
        
        Args:
            focus: Current exhibit focus index
            dwell: Current dwell time (engagement signal)
            utterance: User's utterance text
            response_type: Simulator's response type (question, statement, confusion, etc.)
            simulator: Simulator instance for state synchronization
        """
        if focus is not None:
            self.focus = focus
        if dwell is not None:
            self.dwell = dwell
        if utterance is not None:
            self.last_user_utterance = utterance
            self._update_dialogue_history("user", utterance)

            # Track user intent for responsiveness checking
            if utterance.strip():
                recognizer = get_dialoguebert_recognizer()
                self._last_user_intent = recognizer.classify_intent_category(utterance)
        
        # Track response type for responsiveness reward calculation
        if response_type is not None:
            self._last_user_response_type = response_type

        # NEW: Sync simulator with state information for transitions
        if simulator and hasattr(simulator, 'update_from_state'):
            # Pass current focus and any target exhibit from transition logic
            target_exhibit = None
            if self.focus > 0 and self.focus <= len(self.exhibit_keys):
                target_exhibit = self.exhibit_keys[self.focus - 1]

            simulator.update_from_state(self.focus, target_exhibit)
    



    def add_dialoguebert_insights_to_info(self, info: Dict[str, Any]):
        """Add DialogueBERT insights to info dict after user state is updated"""
        try:
            def _cos(a, b):
                an = float(np.linalg.norm(a) + 1e-8)
                bn = float(np.linalg.norm(b) + 1e-8)
                return float(np.dot(a, b) / (an * bn)) if an > 0.0 and bn > 0.0 else 0.0

            # Re-compute embeddings with updated user utterance
            recognizer = get_dialoguebert_recognizer()
            
            # Store previous embeddings
            prev_intent = getattr(self, '_last_intent_embedding', np.zeros(768, dtype=np.float32))
            prev_context = getattr(self, '_last_dialogue_context', np.zeros(768, dtype=np.float32))
            self._prev_intent_embedding = prev_intent.astype(np.float32)
            self._prev_dialogue_context = prev_context.astype(np.float32)
            
            # Compute new embeddings with current utterance
            intent_embedding = recognizer.get_intent_embedding(
                self.last_user_utterance, role="user"
            )
            # dialogue_history already contains (role, utterance) tuples
            dialogue_context = recognizer.get_dialogue_context(
                self.dialogue_history, max_turns=3
            )
            
            # Update stored embeddings
            self._last_intent_embedding = intent_embedding.astype(np.float32)
            self._last_dialogue_context = dialogue_context.astype(np.float32)
            
            # Compute insights
            intent_category = recognizer.classify_intent_category(self.last_user_utterance)
            intent_norm = float(np.linalg.norm(intent_embedding))
            context_norm = float(np.linalg.norm(dialogue_context))
            cosine_intent_context = _cos(intent_embedding, dialogue_context)
            cosine_intent_prev = _cos(intent_embedding, prev_intent)
            cosine_context_prev = _cos(dialogue_context, prev_context)

            info["dialoguebert_insights"] = {
                "intent_category": intent_category,
                "intent_norm": intent_norm,
                "context_norm": context_norm,
                "cosine_intent_context": cosine_intent_context,
                "cosine_intent_prev": cosine_intent_prev,
                "cosine_context_prev": cosine_context_prev
            }
        except Exception:
            # Insights are optional; ignore failures
            pass
