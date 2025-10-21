"""
Sim8 Simulator Adapter for HRL Museum Agent

This adapter implements a production-friendly subset of the sim8_1 notebook logic:
- AOI detection via sentence-transformers (keyword + semantic fallback)
- Persona-aware response types (question, confusion, reference, statement, silence)
- Gaze feature synthesis with dwell time as reward signal
- Clean API: initialize_session, get_current_aoi, generate_user_response, get_current_state

Design goals:
- No Colab, HF login, or large LLM dependencies
- Deterministic-ish behavior with randomness for diversity
- Compatible with existing training loop and environment
"""

import random
import re
from typing import Dict, Any, List, Optional, Tuple

import numpy as np

try:
    from sentence_transformers import SentenceTransformer, util  # Optional but recommended
    _has_st = True
except Exception:
    _has_st = False
    SentenceTransformer = None
    util = None


class Sim8Simulator:
    """Simulator with AOI detection, persona behavior, and gaze synthesis."""

    AOI_TO_PARENT = {
        "King Caspar": "C6", "Incense Pot": "C6", "Necklace": "C6", "Earring": "C6",
        "Ring": "C6", "Doublet": "C6", "Gemstones": "C6",
        "Turban": "C5", "White ostrich feather": "C5", "Blue garment": "C5", "Young boy": "C5",
        "Red ostrich feather": "B2", "Cavalier hat": "B2", "Gilt garment": "B2", "Dom Miguel": "B2",
        "Ivory tusk": "B3", "Pedro Sunda": "B3", "Cloth": "B3",
        "Box": "B1", "Diego Bemba": "B1", "Clothes": "B1"
    }

    PARENT_TO_EXHIBIT = {
        "C6": "King_Caspar",
        "C5": "Turban",
        "B2": "Cavalier_hat",
        "B3": "Ivory_tusk",
        "B1": "Box",
    }

    # Additional mapping for AOIs that are treated as separate exhibits
    AOI_TO_EXHIBIT = {
        "Gemstones": "Gemstones",
        "Necklace": "Necklace",
        "Ring": "Ring",
        "Doublet": "Doublet",
        "Incense_Pot": "Incense_Pot",
        "Earring": "Earring",
        "Red_ostrich_feather": "Red_ostrich_feather",
        "Gilt_garment": "Gilt_garment",
        "Pedro_Sunda": "Pedro_Sunda",
        "Diego_Bemba": "Diego_Bemba",
        "White_ostrich_feather": "White_ostrich_feather",
        "Blue_garment": "Blue_garment",
        "Young_boy": "Young_boy",
        "Cloth": "Cloth",
        "Clothes": "Clothes",
    }

    PERSONAS = ["Agreeable", "Conscientious", "Neurotic"]

    # Gaze feature labels - FIRST FEATURE IS DWELL TIME FOR REWARD
    GAZE_LABELS = [
        "DwellTime", "SaccadeSpan", "TurnGazeEntropy",
        "TurnFixChangeRate", "DominantObjectRatio", "GazeEntryLatency"
    ]
    SILENCE_STATS = {
        "Agreeable": {"TurnScanpathLength": (78.809, 132.598), "SaccadeSpan": (0.1030, 0.0556),
                       "TurnGazeEntropy": (0.8501, 0.4581), "TurnFixChangeRate": (2.1581, 0.8471),
                       "DominantObjectRatio": (0.7233, 0.1832), "GazeEntryLatency": (6.4418, 12.5257)},
        "Conscientious": {"TurnScanpathLength": (49.190, 60.532), "SaccadeSpan": (0.1224, 0.0673),
                           "TurnGazeEntropy": (0.5848, 0.5409), "TurnFixChangeRate": (2.0858, 1.4099),
                           "DominantObjectRatio": (0.7938, 0.2126), "GazeEntryLatency": (4.6317, 6.6963)},
        "Neurotic": {"TurnScanpathLength": (41.272, 49.602), "SaccadeSpan": (0.1249, 0.0652),
                      "TurnGazeEntropy": (1.0182, 0.3763), "TurnFixChangeRate": (2.8544, 0.7449),
                      "DominantObjectRatio": (0.6576, 0.2000), "GazeEntryLatency": (2.3105, 3.7783)}
    }

    def __init__(self, exhibits: Optional[List[str]] = None, seed: int = 42):
        self.rng = random.Random(seed)

        # If exhibits provided, use them; else derive from mappings
        if exhibits:
            self.exhibits = exhibits
        else:
            # Include both parent exhibits and AOI exhibits
            parent_exhibits = list(set(self.PARENT_TO_EXHIBIT.values()))
            aoi_exhibits = list(set(self.AOI_TO_EXHIBIT.values()))
            self.exhibits = list(set(parent_exhibits + aoi_exhibits))

        # Sentence transformer model
        self._st_model = None
        self._aoi_list = list(self.AOI_TO_PARENT.keys())
        self._aoi_embeddings = None
        if _has_st:
            try:
                self._st_model = SentenceTransformer("all-MiniLM-L6-v2")
                self._aoi_embeddings = self._st_model.encode(self._aoi_list, convert_to_tensor=True)
            except Exception:
                self._st_model = None
                self._aoi_embeddings = None

        # Session state
        self.current_persona: Optional[str] = None
        self.current_exhibit: Optional[str] = None
        self.current_aoi: Optional[str] = None
        self.aoi_usage_count: Dict[str, int] = {}
        self.seen_aois: set = set()
        self.consecutive_silence_count: int = 0
        self.last_user_response: Dict[str, Any] = {}
        
        # Track conversation context for realistic responses
        self.last_user_question: Optional[str] = None  # What user asked
        self.last_agent_utterance: Optional[str] = None  # What agent said
        self.conversation_flow: List[str] = []  # Track conversation quality
        
        # NEW: Museum context for grounding user responses
        self.museum_context = self._build_museum_context()
        
        # NEW: Disengagement tracking
        self.engagement_level = 1.0  # 0.0 = fully disengaged, 1.0 = fully engaged
        self.off_topic_strikes = 0  # Track consecutive off-topic responses
    
    def _build_museum_context(self) -> Dict[str, Any]:
        """Build knowledge of museum structure for context-aware responses"""
        context = {
            "exhibits": {},
            "aoi_descriptions": {}
        }
        
        # Map exhibits to their AOIs
        for aoi, parent_code in self.AOI_TO_PARENT.items():
            exhibit_name = self.PARENT_TO_EXHIBIT.get(parent_code, "Unknown")
            if exhibit_name not in context["exhibits"]:
                context["exhibits"][exhibit_name] = []
            context["exhibits"][exhibit_name].append(aoi)
            
            # Simple descriptions for AOIs
            context["aoi_descriptions"][aoi] = f"{aoi} from the {exhibit_name} exhibit"
        
        return context

    # ===== Public API =====
    def initialize_session(self, persona: Optional[str] = None):
        self.current_persona = persona or self.rng.choice(self.PERSONAS)
        self.current_exhibit = self.rng.choice(self.exhibits)
        self.current_aoi = self._pick_initial_aoi(self.current_exhibit)
        self.aoi_usage_count.clear()
        self.seen_aois.clear()
        self.consecutive_silence_count = 0
        self.last_user_response = {}
        
        # Initialize dialogue history and learning
        self.dialogue_history = []
        self.facts_learned = set()
        self.exhibits_visited = set()
        self.max_history_length = 8

    def get_current_aoi(self) -> str:
        """Return current exhibit (for env focus) to match previous simulator contract."""
        return self.current_exhibit or self.exhibits[0]

    def generate_user_response(self, agent_utterance: str) -> Dict[str, Any]:
        """Generate a response dict with utterance, aoi, persona, gaze_features, response_type."""
        # Store agent's utterance for context tracking
        self.last_agent_utterance = agent_utterance
        
        # VERY RARE silence (1% only, prevents Turn 2 silence issue)
        # Only in first turn OR if too many consecutive silences
        if self.consecutive_silence_count >= 2:
            # Force non-silence to prevent dialogue breakdown
            pass
        elif len(self.dialogue_history) < 2 and self.rng.random() < 0.01:
            # Only in very first turn, 1% chance
            self.consecutive_silence_count += 1
            response = self._make_silence_response()
            self.last_user_response = response
            return response
        else:
            # Never silence after first turn
            pass

        # Reset silence streak on any non-silence response
        self.consecutive_silence_count = 0

        # STEP 0: Check agent relevance to current context
        relevance_score = self._check_agent_relevance(agent_utterance)
        
        # Track agent utterance in dialogue history
        self.dialogue_history.append({"role": "agent", "text": agent_utterance})
        if len(self.dialogue_history) > self.max_history_length:
            self.dialogue_history.pop(0)
        
        # STEP 1: Check if agent answered our previous question
        answered_well = self._check_if_question_answered(agent_utterance)
        
        # STEP 2: Detect if agent is suggesting movement (higher chance to switch)
        is_suggesting_move = self._is_suggesting_movement(agent_utterance)
        
        # STEP 3: Detect AOI from agent utterance; fallback to current
        detected_aoi, parent = self._detect_aoi_and_parent(agent_utterance)
        
        # If agent detected a new exhibit/AOI, follow suggestion (deterministic for transitions)
        if detected_aoi is not None and detected_aoi != self.current_aoi:
            # Always follow transition suggestions (deterministic behavior)
            if is_suggesting_move:
                # Switch to detected AOI for transitions
                pass  # Keep detected_aoi
            else:
                # For non-transition suggestions, use probabilistic behavior but higher chance
                switch_prob = 0.85  # Much higher chance to follow non-transition suggestions
                if self.rng.random() < switch_prob:
                    pass  # Keep detected_aoi
                else:
                    # Stay with current
                    detected_aoi = self.current_aoi
                    parent = self.AOI_TO_PARENT.get(detected_aoi, None)
        else:
            # No new AOI detected, fallback to current
            detected_aoi = self.current_aoi
            parent = self.AOI_TO_PARENT.get(detected_aoi, None)
            
            # Random exploration to other exhibits (lower chance)
            if self.rng.random() < 0.25:
                detected_aoi, parent = self._switch_to_related_aoi(detected_aoi)

        # Update session mapping
        self.current_aoi = detected_aoi
        self.current_exhibit = self.PARENT_TO_EXHIBIT.get(parent, self.current_exhibit)
        self.aoi_usage_count[detected_aoi] = self.aoi_usage_count.get(detected_aoi, 0) + 1
        self.seen_aois.add(detected_aoi)

        # STEP 4: Choose response type based on context and agent quality
        rtype = self._determine_response_type_contextual(agent_utterance, answered_well)
        
        # STEP 5: Generate utterance with LLM-guided response (NEW!)
        import time
        sim_start = time.time()
        utterance = self._synthesize_llm_guided_utterance(agent_utterance, rtype, detected_aoi, answered_well)
        self._last_sim_llm_time = time.time() - sim_start
        
        # STEP 6: Generate gaze features based on response quality
        gaze_features = self._synthesize_contextual_gaze(rtype, answered_well)
        
        # Store user's question if they asked one
        if rtype == "question":
            self.last_user_question = utterance

        response = {
            "utterance": utterance,
            "aoi": detected_aoi,
            "persona": self.current_persona,
            "gaze_features": gaze_features,
            "response_type": rtype,
            "answered_well": answered_well,  # Track for debugging
            "simulator_llm_time": getattr(self, '_last_sim_llm_time', 0.0)
        }
        self.last_user_response = response
        
        # Track in memory (NEW!)
        self.exhibits_visited.add(self.current_exhibit)
        self.dialogue_history.append({"role": "user", "text": utterance or ""})
        
        # Extract fact IDs if present
        fact_ids = re.findall(r'\[([A-Z]{2}_\d{3})\]', agent_utterance)
        for fact_id in fact_ids:
            self.facts_learned.add(fact_id)
        
        return response

    def get_current_state(self) -> Dict[str, Any]:
        return {
            "aoi": self.current_aoi,
            "current_exhibit": self.current_exhibit,
            "persona": self.current_persona,
            "seen_aois": list(self.seen_aois),
            "aoi_usage_count": dict(self.aoi_usage_count),
            "consecutive_silence_count": self.consecutive_silence_count,
            "last_user_response": dict(self.last_user_response) if self.last_user_response else {}
        }

    def update_from_state(self, state_focus: int, target_exhibit: str = None):
        """Update simulator state based on environment state information"""
        # If we have a target exhibit from transition logic, prioritize it
        if target_exhibit and target_exhibit != self.current_exhibit:
            # Find an AOI for the target exhibit
            for parent_code, parent_exhibit in self.PARENT_TO_EXHIBIT.items():
                if parent_exhibit == target_exhibit:
                    candidate_aois = [a for a, p in self.AOI_TO_PARENT.items() if p == parent_code]
                    if candidate_aois:
                        new_aoi = self.rng.choice(candidate_aois)
                        self.current_aoi = new_aoi
                        self.current_exhibit = target_exhibit
                        self.aoi_usage_count[new_aoi] = self.aoi_usage_count.get(new_aoi, 0) + 1
                        self.seen_aois.add(new_aoi)
                        print(f"STATE INFLUENCE: Transitioned to {target_exhibit} (AOI: {new_aoi})")
                        return

        # Otherwise, use focus from state if it's different from current
        if state_focus > 0 and state_focus <= len(self.exhibits):
            target_exhibit = self.exhibits[state_focus - 1]
            if target_exhibit != self.current_exhibit:
                # Find an AOI for the target exhibit
                for parent_code, parent_exhibit in self.PARENT_TO_EXHIBIT.items():
                    if parent_exhibit == target_exhibit:
                        candidate_aois = [a for a, p in self.AOI_TO_PARENT.items() if p == parent_code]
                        if candidate_aois:
                            new_aoi = self.rng.choice(candidate_aois)
                            self.current_aoi = new_aoi
                            self.current_exhibit = target_exhibit
                            self.aoi_usage_count[new_aoi] = self.aoi_usage_count.get(new_aoi, 0) + 1
                            self.seen_aois.add(new_aoi)
                            print(f"STATE SYNC: Updated to {target_exhibit} (AOI: {new_aoi})")

    # ===== Internals =====
    def _pick_initial_aoi(self, exhibit: str) -> str:
        parent = None
        for k, v in self.PARENT_TO_EXHIBIT.items():
            if v == exhibit:
                parent = k
                break
        candidates = [a for a, p in self.AOI_TO_PARENT.items() if p == parent]
        return self.rng.choice(candidates) if candidates else self.rng.choice(list(self.AOI_TO_PARENT.keys()))

    def _detect_aoi_and_parent(self, text: str, sim_threshold: float = 0.35) -> Tuple[Optional[str], Optional[str]]:
        t = (text or "").lower()

        # First, try exhibit names (for cross-exhibit transitions) - improved matching
        # Check both PARENT_TO_EXHIBIT and AOI_TO_EXHIBIT mappings
        all_exhibit_names = set(self.PARENT_TO_EXHIBIT.values()) | set(self.AOI_TO_EXHIBIT.values())

        for exhibit_name in all_exhibit_names:
            # Remove underscores and check for exhibit name in text (more flexible matching)
            exhibit_clean = exhibit_name.replace("_", " ").lower()

            # Direct match
            if exhibit_clean in t:
                # Find an AOI for this exhibit
                if exhibit_name in self.AOI_TO_EXHIBIT:
                    # It's a direct AOI-to-exhibit mapping
                    aoi = exhibit_name
                    parent_code = self.AOI_TO_PARENT.get(aoi, "C6")  # Default to C6 if not found
                    return aoi, parent_code
                else:
                    # It's a parent exhibit, find an AOI within it
                    for parent_code, parent_exhibit in self.PARENT_TO_EXHIBIT.items():
                        if parent_exhibit == exhibit_name:
                            # Get random AOI from this exhibit
                            candidate_aois = [a for a, p in self.AOI_TO_PARENT.items() if p == parent_code]
                            if candidate_aois:
                                aoi = self.rng.choice(candidate_aois)
                                return aoi, parent_code

            # Partial word matching for exhibit names (e.g., "caspar" matches "King Caspar")
            exhibit_words = exhibit_clean.split()
            for word in exhibit_words:
                if len(word) > 3 and word in t:  # Only match longer words to avoid false positives
                    if exhibit_name in self.AOI_TO_EXHIBIT:
                        # It's a direct AOI-to-exhibit mapping
                        aoi = exhibit_name
                        parent_code = self.AOI_TO_PARENT.get(aoi, "C6")
                        return aoi, parent_code
                    else:
                        # It's a parent exhibit
                        for parent_code, parent_exhibit in self.PARENT_TO_EXHIBIT.items():
                            if parent_exhibit == exhibit_name:
                                candidate_aois = [a for a, p in self.AOI_TO_PARENT.items() if p == parent_code]
                                if candidate_aois:
                                    aoi = self.rng.choice(candidate_aois)
                                    return aoi, parent_code

        # Keyword match for AOIs (handle possessives like "caspar's")
        for aoi in self._aoi_list:
            parts = aoi.lower().split()
            for part in parts:
                # Match with optional possessive 's
                if re.search(rf"\b{re.escape(part)}('s)?\b", t):
                    return aoi, self.AOI_TO_PARENT[aoi]

        # Enhanced semantic fallback for transition contexts
        if self._st_model is not None and self._aoi_embeddings is not None:
            try:
                query_emb = self._st_model.encode(text, convert_to_tensor=True)
                from torch import max as tmax  # local import to avoid hard dep if torch missing
                cos_scores = util.cos_sim(query_emb, self._aoi_embeddings)[0]
                top_score, top_idx = tmax(cos_scores, dim=0)

                # Lower threshold for transition contexts (agent suggesting movement)
                transition_keywords = ["visit", "see", "check out", "explore", "move", "next", "let's", "shall we"]
                is_transition_context = any(keyword in t for keyword in transition_keywords)
                effective_threshold = sim_threshold * 0.7 if is_transition_context else sim_threshold

                if float(top_score) >= effective_threshold:
                    aoi = self._aoi_list[int(top_idx)]
                    return aoi, self.AOI_TO_PARENT[aoi]
            except Exception:
                pass
        return None, None

    def _switch_to_related_aoi(self, current_aoi: str) -> Tuple[str, Optional[str]]:
        """Switch to a related AOI (70% same exhibit, 30% different exhibit)"""
        if self.rng.random() < 0.7:
            # Switch within same exhibit (sibling AOIs)
            parent = self.AOI_TO_PARENT.get(current_aoi)
            siblings = [a for a, p in self.AOI_TO_PARENT.items() if p == parent and a != current_aoi]
            if siblings:
                choice = self.rng.choice(siblings)
                return choice, self.AOI_TO_PARENT[choice]
        
        # Switch to different exhibit
        current_parent = self.AOI_TO_PARENT.get(current_aoi)
        other_parents = [p for p in self.PARENT_TO_EXHIBIT.keys() if p != current_parent]
        if other_parents:
            new_parent = self.rng.choice(other_parents)
            new_aois = [a for a, p in self.AOI_TO_PARENT.items() if p == new_parent]
            if new_aois:
                choice = self.rng.choice(new_aois)
                return choice, new_parent
        
        # Fallback: stay with current
        return current_aoi, current_parent

    def _is_suggesting_movement(self, text: str) -> bool:
        """Detect if agent is suggesting moving to another exhibit"""
        text_lower = (text or "").lower()
        move_keywords = [
            "check out", "explore", "visit", "see", "move", "head to",
            "would you like to", "shall we", "let's", "next", "another",
            "recommend", "suggest", "ready to", "finished", "done"
        ]
        return any(keyword in text_lower for keyword in move_keywords)
    
    def _detect_hallucinations(self, agent_utterance: str) -> tuple:
        """Detect if agent claims conversations that didn't happen."""
        agent_lower = (agent_utterance or "").lower()
        
        # Detect memory-seeking language
        memory_patterns = ["recall", "remember", "discussed", "we talked about", "earlier", "before"]
        has_memory_claim = any(pattern in agent_lower for pattern in memory_patterns)
        
        if not has_memory_claim:
            return False, ""
        
        # If claiming past discussion but dialogue history is empty = hallucination!
        if not self.dialogue_history:
            return True, "first_turn_memory_claim"
        
        # Extract claimed topic from utterance
        import re
        topic_patterns = [
            r'(?:about|regarding|on)\s+([a-z\s]+?)(?:\?|\.)',
            r'our\s+(?:discussion|conversation)\s+([a-z\s]+?)(?:\?|\.)',
        ]
        
        claimed_topic = None
        for pattern in topic_patterns:
            match = re.search(pattern, agent_lower)
            if match:
                claimed_topic = match.group(1).strip()
                break
        
        if claimed_topic:
            # Search dialogue history for this topic
            history_text = " ".join([turn.get("text", "").lower() for turn in self.dialogue_history])
            if claimed_topic not in history_text:
                return True, f"topic_not_discussed_{claimed_topic}"
        
        return False, ""

    def _check_agent_relevance(self, agent_utterance: str) -> float:
        """
        Check if agent's response is relevant to current context.
        Returns relevance score 0.0-1.0 and updates engagement metric.
        
        Checks:
        - Is agent talking about current AOI/exhibit?
        - Is agent using meta-commentary (bad)?
        - Is agent being natural vs robotic?
        """
        relevance_score = 1.0
        agent_lower = (agent_utterance or "").lower()
        
        # CHECK 0: Hallucination detection (NEW!)
        is_hallucinating, hallucination_reason = self._detect_hallucinations(agent_utterance)
        if is_hallucinating:
            relevance_score -= 0.5  # MAJOR PENALTY
            self.off_topic_strikes += 2  # Double strike
            self.engagement_level *= 0.4  # Significant drop
        
        # Check current AOI mention
        current_aoi_clean = self.current_aoi.lower().replace("_", " ") if self.current_aoi else ""
        current_exhibit_clean = self.current_exhibit.lower().replace("_", " ") if self.current_exhibit else ""
        
        aoi_mentioned = current_aoi_clean in agent_lower
        exhibit_mentioned = current_exhibit_clean in agent_lower
        
        if not (aoi_mentioned or exhibit_mentioned):
            relevance_score -= 0.3  # Penalty: not mentioning current focus
        
        # Check for meta-commentary (BAD: "here's a response", "visitor:", "guide:")
        meta_patterns = ["here's a response", "here is a response", "the guide says", 
                        "visitor:", "guide:", "museum guide:", "assistant:"]
        meta_count = sum(1 for pattern in meta_patterns if pattern in agent_lower)
        if meta_count > 0:
            relevance_score -= 0.2 * meta_count
        
        # Check for natural language markers (GOOD)
        natural_markers = ["actually", "honestly", "you know", "interesting", "fascinating"]
        natural_count = sum(1 for marker in natural_markers if marker in agent_lower)
        relevance_score += 0.1 * min(natural_count, 2)
        
        # Clamp score
        relevance_score = max(0.0, min(1.0, relevance_score))
        
        # Update disengagement metric
        if relevance_score < 0.5:
            self.off_topic_strikes += 1
            if self.off_topic_strikes >= 2:
                self.engagement_level *= 0.6  # Drop engagement significantly
        else:
            self.off_topic_strikes = 0
            self.engagement_level = min(1.0, self.engagement_level + 0.05)
        
        return relevance_score
    
    def _check_if_question_answered(self, agent_utterance: str) -> bool:
        """Check if agent actually answered the user's previous question"""
        if not self.last_user_question:
            return True  # No previous question to answer
        
        # Simple heuristic: check if agent provided substantive response
        agent_lower = (agent_utterance or "").lower()
        
        # Bad signs: agent just asks back without answering
        if agent_lower.count("?") >= 2 and len(agent_lower.split()) < 30:
            return False  # Agent mostly asked questions, didn't answer
        
        # Good signs: agent provided information
        info_keywords = ["is", "was", "are", "were", "made", "created", "symbolize", "represent", 
                        "used", "from", "in", "dates", "century", "period"]
        has_info = sum(1 for kw in info_keywords if kw in agent_lower) >= 2
        
        # Check if agent mentioned what user asked about
        if self.last_user_question:
            question_topic = self._extract_topic_from_question(self.last_user_question)
            if question_topic and question_topic.lower() in agent_lower:
                has_info = True
        
        return has_info
    
    def _extract_topic_from_question(self, question: str) -> Optional[str]:
        """Extract main topic from user's question"""
        import re
        # Extract noun phrases after "about the" or "of the"
        match = re.search(r'(?:about|of) the ([a-zA-Z\s]+)', question.lower())
        if match:
            return match.group(1).strip()
        return None
    
    def _determine_response_type_contextual(self, agent_utterance: str, answered_well: bool) -> str:
        """Determine response type based on agent's utterance quality"""
        text = (agent_utterance or "").lower()
        
        # If agent answered our question well, show satisfaction or ask follow-up
        if answered_well and self.last_user_question:
            if self.rng.random() < 0.4:
                return "acknowledgment"  # "That's interesting!"
            elif self.rng.random() < 0.6:
                return "follow_up_question"  # Related question
            else:
                return "statement"  # Show engagement
        
        # If agent didn't answer well, express confusion or re-ask
        if not answered_well and self.last_user_question:
            if self.rng.random() < 0.5:
                return "confusion"  # "I'm not sure I understand..."
            else:
                return "question"  # Ask a different question
        
        # Agent asked a question - respond appropriately
        if "?" in text:
            if self.rng.random() < 0.7:
                return "question"  # Ask own question
            else:
                return "statement"  # Make statement
        
        # Neurotic persona gets confused sometimes
        if self.current_persona == "Neurotic" and self.rng.random() < 0.15:
            return "confusion"
        
        # Otherwise, varied responses
        rand = self.rng.random()
        if rand < 0.5:
            return "question"
        elif rand < 0.7:
            return "statement"
        elif rand < 0.85:
            return "acknowledgment"
        else:
            return "follow_up_question"

    def _synthesize_contextual_utterance(self, rtype: str, aoi: str, answered_well: bool) -> str:
        """Create a plausible utterance for the given response_type and AOI."""
        if rtype == "silence":
            return ""
        
        # Clean AOI name for better templates
        clean_aoi = aoi.replace("_", " ")
        
        # Question templates - much more variety
        question_templates = [
            f"What does the {clean_aoi} signify?",
            f"Can you tell me more about the {clean_aoi}?",
            f"Why is the {clean_aoi} important?",
            f"What's special about the {clean_aoi}?",
            f"Who created the {clean_aoi}?",
            f"When was the {clean_aoi} made?",
            f"Where did the {clean_aoi} come from?",
            f"How was the {clean_aoi} used?",
            f"What's the story behind the {clean_aoi}?",
            f"What materials is the {clean_aoi} made from?",
            f"What's the significance of the {clean_aoi}?",
            f"Is there symbolism in the {clean_aoi}?",
            f"What culture does the {clean_aoi} represent?",
            f"Tell me about the history of the {clean_aoi}.",
            f"What period is the {clean_aoi} from?",
        ]
        
        # Statement templates - more natural
        statement_templates = [
            f"That's interesting about the {clean_aoi}.",
            f"I like the {clean_aoi}.",
            f"The {clean_aoi} looks beautiful.",
            f"I find the {clean_aoi} fascinating.",
            f"The {clean_aoi} is quite striking.",
            f"I'm drawn to the {clean_aoi}.",
            f"The craftsmanship of the {clean_aoi} is impressive.",
            f"I appreciate the detail in the {clean_aoi}.",
            f"The {clean_aoi} has such rich colors.",
            f"The {clean_aoi} stands out to me.",
        ]
        
        # Reference templates (referring back)
        reference_templates = [
            f"You mentioned the {clean_aoi} earlier.",
            f"Going back to the {clean_aoi}...",
            f"About the {clean_aoi}...",
            f"Regarding the {clean_aoi}...",
            f"I was thinking about the {clean_aoi}.",
            f"Can we return to the {clean_aoi}?",
            f"I have another question about the {clean_aoi}.",
        ]
        
        # Confusion templates
        confusion_templates = [
            f"I'm not sure I understand about the {clean_aoi}.",
            f"Could you clarify about the {clean_aoi}?",
            f"I'm confused about the {clean_aoi}.",
            f"What did you mean about the {clean_aoi}?",
            f"I didn't quite follow that about the {clean_aoi}.",
            f"Could you explain the {clean_aoi} again?",
        ]
        
        # NEW: Acknowledgment templates (when agent answers well)
        acknowledgment_templates = [
            f"Oh, that's fascinating about the {clean_aoi}!",
            f"I see, that makes sense about the {clean_aoi}.",
            f"That's really interesting, thank you!",
            f"Wow, I didn't know that about the {clean_aoi}.",
            f"That's helpful, I understand better now.",
            f"Interesting perspective on the {clean_aoi}.",
            f"Thank you for explaining that!",
            f"That clears things up about the {clean_aoi}.",
        ]
        
        # NEW: Follow-up question templates (when engaged and curious)
        followup_templates = [
            f"What else can you tell me about the {clean_aoi}?",
            f"How does that relate to other pieces in the collection?",
            f"Can you elaborate on that?",
            f"What's the historical context for this?",
            f"Are there other similar {clean_aoi} pieces?",
            f"What influenced the artist's choice here?",
        ]
        
        if rtype == "acknowledgment":
            return self.rng.choice(acknowledgment_templates)
        elif rtype == "follow_up_question":
            return self.rng.choice(followup_templates)
        elif rtype == "question":
            return self.rng.choice(question_templates)
        elif rtype == "statement":
            return self.rng.choice(statement_templates)
        elif rtype == "reference":
            return self.rng.choice(reference_templates)
        elif rtype == "confusion":
            return self.rng.choice(confusion_templates)
        else:
            return self.rng.choice(question_templates)

    def _synthesize_llm_guided_utterance(self, agent_utterance: str, rtype: str, aoi: str, answered_well: bool) -> str:
        """
        Generate user response using LLM guidance to ensure responses directly address agent utterances.
        Falls back to templates if LLM is unavailable.
        """
        try:
            # Try to use LLM for guided response - uses simulator LLM from centralized config
            from LLM_CONFIG import get_simulator_llm
            import os
            
            # Skip if in fast mode (use templates only)
            if os.environ.get('HRL_FAST_MODE') == '1':
                return self._synthesize_contextual_utterance(rtype, aoi, answered_well)
            
            import time
            start_time = time.time()
            print(f"[Simulator LLM] Generating user response ({rtype})...", flush=True)
            llm = get_simulator_llm()
            clean_aoi = aoi.replace("_", " ")
            
            # Build a prompt that guides the LLM to generate contextually appropriate responses
            system_prompt = f"""You are a museum visitor with persona: {self.current_persona}.
Generate a SHORT (1-2 sentence) NATURAL response to what the museum guide just said.
Be conversational and genuine. React specifically to their statement.
Response type should be: {rtype}
Current exhibit: {clean_aoi}"""

            user_prompt = f"""Guide said: "{agent_utterance}"

Your {rtype} response (1-2 sentences):"""

            response = llm.generate(user_prompt, system_prompt=system_prompt)
            elapsed = time.time() - start_time
            print(f"[Simulator LLM] User response received in {elapsed:.2f}s ({len(response)} chars)", flush=True)
            response = response.strip().strip('"')
            return response[:300]  # Limit length
            
        except Exception:
            # Fallback to template-based approach
            return self._synthesize_contextual_utterance(rtype, aoi, answered_well)

    def _synthesize_contextual_gaze(self, rtype: str, answered_well: bool) -> List[float]:
        """Generate synthetic gaze features based on response type AND answer quality."""
        # CRITICAL: Adjust engagement based on whether agent answered well!
        
        # Base engagement modifiers
        engagement_bonus = 0.0
        if answered_well:
            engagement_bonus = 0.2  # Boost engagement when agent is helpful
        else:
            engagement_bonus = -0.25  # Reduce engagement when agent is off-topic
        
        # Apply engagement level multiplier from disengagement tracking
        engagement_bonus *= self.engagement_level
        
        # Different patterns for different response types
        if rtype in ["acknowledgment", "follow_up_question"]:
            # HIGH engagement - agent answered well, user is satisfied and curious
            dwell_time = self._clip(self._randf(0.75, 0.95) + engagement_bonus, 0.2, 1.0)
            saccade_span = max(0.05, np.random.normal(0.07, 0.03))  # Low saccades (focused)
        
        elif rtype == "question":
            # Engagement depends on context - high if genuinely curious, lower if confused
            base_dwell = self._randf(0.6, 0.9)
            dwell_time = self._clip(base_dwell + engagement_bonus, 0.2, 1.0)
            saccade_span = max(0.05, np.random.normal(0.08, 0.04))
        
        elif rtype == "statement":
            # Moderate engagement when making statements
            base_dwell = self._randf(0.5, 0.8)
            dwell_time = self._clip(base_dwell + engagement_bonus, 0.2, 1.0)
            saccade_span = max(0.05, np.random.normal(0.09, 0.04))
        
        elif rtype == "confusion":
            # LOW engagement when confused - agent was unhelpful
            dwell_time = self._clip(self._randf(0.25, 0.50) + engagement_bonus, 0.1, 0.6)
            saccade_span = max(0.05, np.random.normal(0.12, 0.05))  # Higher saccades (unfocused)
        
        else:
            # Default moderate engagement
            dwell_time = self._clip(self._randf(0.5, 0.8), 0.2, 1.0)
            saccade_span = max(0.05, np.random.normal(0.09, 0.04))
        
        # Common gaze features (persona-influenced)
        persona = self.current_persona or "Agreeable"
        stats = self.SILENCE_STATS.get(persona, self.SILENCE_STATS["Agreeable"])
        
        gaze_entropy = self._clip(np.random.normal(stats["TurnGazeEntropy"][0], stats["TurnGazeEntropy"][1]), 0.0, 2.5)
        fix_change_rate = self._clip(np.random.normal(stats["TurnFixChangeRate"][0], stats["TurnFixChangeRate"][1]), 0.2, 4.0)
        dom_ratio = self._clip(dwell_time * self._randf(0.6, 0.95), 0.0, 1.0)
        entry_latency = self._clip(np.random.normal(stats["GazeEntryLatency"][0], stats["GazeEntryLatency"][1]), 0.1, 12.0)

        return [
            float(dwell_time),
            float(saccade_span),
            float(gaze_entropy),
            float(fix_change_rate),
            float(dom_ratio),
            float(entry_latency),
        ]

    def _make_silence_response(self) -> Dict[str, Any]:
        persona = self.current_persona or "Agreeable"
        
        # For silence, low engagement (dwell time should be low)
        # Sample other gaze features from persona stats
        dwell_time = self._randf(0.1, 0.4)  # Low engagement during silence
        
        # Sample other features from stats
        stats = self.SILENCE_STATS[persona]
        saccade_span = max(0.05, np.random.normal(0.1, 0.05))  # Higher saccades during silence
        gaze_entropy = self._clip(np.random.normal(stats["TurnGazeEntropy"][0], stats["TurnGazeEntropy"][1]), 0.0, 2.5)
        fix_change_rate = self._clip(np.random.normal(stats["TurnFixChangeRate"][0], stats["TurnFixChangeRate"][1]), 0.2, 4.0)
        dom_ratio = self._clip(self._randf(0.4, 0.7), 0.0, 1.0)  # Lower dominance during silence
        entry_latency = self._clip(np.random.normal(stats["GazeEntryLatency"][0], stats["GazeEntryLatency"][1]), 0.1, 12.0)
        
        feats = [
            float(dwell_time),
            float(saccade_span),
            float(gaze_entropy),
            float(fix_change_rate),
            float(dom_ratio),
            float(entry_latency),
        ]
        
        return {
            "utterance": None,
            "aoi": self.current_aoi,
            "persona": persona,
            "gaze_features": feats,
            "response_type": "silence",
        }

    def _clip(self, v: float, lo: float, hi: float) -> float:
        return float(max(lo, min(hi, v)))

    def _randf(self, lo: float, hi: float) -> float:
        return lo + (hi - lo) * self.rng.random()
