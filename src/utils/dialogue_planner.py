"""
Dialogue Planner for HRL Museum Dialogue Agent

This module implements structured prompting for LLM-based dialogue generation in the
HRL museum agent. It translates high-level options and subactions into structured
prompts that guide LLM generation while maintaining dialogue coherence and policy
adherence.
"""

from typing import List, Optional


def build_prompt(option: str, subaction: str, ex_id: Optional[str], 
                last_utt: str, facts_all: List[str], facts_used: List[str], 
                selected_fact: Optional[str], dialogue_history: List[str] = None,
                exhibit_names: List[str] = None, knowledge_graph=None,
                target_exhibit: str = None, coverage_dict: dict = None) -> str:
    """Build structured prompt for LLM dialogue generation"""
    
    # Build context - show facts ONLY for Explain option and Conclude option
    show_facts = (option == "Explain") or (option == "Conclude" and subaction == "SummarizeKeyPoints")
    context_section = _build_enhanced_context_section(ex_id, last_utt, facts_all, facts_used, dialogue_history, exhibit_names, knowledge_graph, show_facts=show_facts)

    # Calculate current exhibit completion for contextual prompts
    current_completion = 0.0
    if coverage_dict and ex_id:
        current_completion = coverage_dict.get(ex_id, {"coverage": 0.0})["coverage"]
    
    # Route to specific subaction function
    if option == "Explain":
        if subaction == "ExplainNewFact":
            return build_explain_new_fact_prompt(ex_id, context_section, facts_all, facts_used, selected_fact, current_completion)
        elif subaction == "RepeatFact":
            return build_repeat_fact_prompt(ex_id, context_section, facts_all, facts_used, selected_fact, current_completion)
        elif subaction == "ClarifyFact":
            return build_clarify_fact_prompt(ex_id, context_section, facts_all, facts_used, selected_fact, current_completion)

    elif option == "AskQuestion":
        if subaction == "AskOpinion":
            return build_ask_opinion_prompt(ex_id, context_section, facts_all, facts_used, current_completion)
        elif subaction == "AskMemory":
            return build_ask_memory_prompt(ex_id, context_section, facts_all, facts_used, current_completion)
        elif subaction == "AskClarification":
            return build_ask_clarification_prompt(ex_id, context_section, facts_all, facts_used, current_completion)

    elif option == "OfferTransition":
        if subaction == "SuggestMove":
            return build_offer_transition_prompt(ex_id, context_section, facts_all, facts_used, exhibit_names, knowledge_graph, target_exhibit, coverage_dict)
        elif subaction == "LinkToOtherExhibit":
            return build_link_to_other_exhibit_prompt(ex_id, context_section, facts_all, facts_used, exhibit_names, knowledge_graph, target_exhibit, coverage_dict, current_completion)

    elif option == "Conclude":
        if subaction == "WrapUp":
            return build_wrap_up_prompt(ex_id, context_section, facts_all, facts_used, current_completion)
        elif subaction == "SummarizeKeyPoints":
            return build_summarize_key_points_prompt(ex_id, context_section, facts_all, facts_used, current_completion)
    
    # Should never reach here - all options should be handled above
    raise ValueError(f"Unknown option '{option}' or subaction '{subaction}'")


def _build_enhanced_context_section(ex_id: Optional[str], last_utt: str, facts_all: List[str], 
                                  facts_used: List[str], dialogue_history: List[str] = None,
                                  exhibit_names: List[str] = None, knowledge_graph=None, show_facts: bool = True) -> str:
    """Build enhanced context section with rich dialogue understanding"""
    context_parts = []
    
    # === EXHIBIT INFORMATION ===
    if ex_id:
        context_parts.append(f"CURRENT EXHIBIT: {ex_id.replace('_', ' ')}")
    
    # === VISITOR'S CURRENT MESSAGE (MOST IMPORTANT) ===
    if last_utt.strip():
        context_parts.append("=" * 60)
        context_parts.append("VISITOR SAID:")
        context_parts.append(f'"{last_utt}"')
        context_parts.append("=" * 60)
        context_parts.append("")
    
    # === DIALOGUE HISTORY (brief) ===
    if dialogue_history and len(dialogue_history) > 0:
        recent_context = dialogue_history[-2:] if len(dialogue_history) > 2 else dialogue_history
        context_parts.append("RECENT CONVERSATION:")
        for i, utterance in enumerate(recent_context, 1):
            context_parts.append(f'  {i}. "{utterance[:80]}..."')
        context_parts.append("")

    # === SHOW FACTS ONLY IF ALLOWED (Explain or Summarize actions) ===
    # NOTE: For ExplainNewFact, facts_all is already filtered to unmentioned facts by env.py
    # The specific prompt builders (build_explain_new_fact_prompt) handle showing facts
    # So we DON'T show facts here to avoid duplication
    if not show_facts:
        # For Ask/Transition/Conclude - NO FACTS SHOWN
        context_parts.append("NO FACTS IN THIS RESPONSE TYPE")
        context_parts.append("Your job is to ask questions or suggest actions ONLY")
        context_parts.append("")
    
    return "\n".join(context_parts)


def _analyze_visitor_utterance(utterance: str) -> str:
    """Analyze visitor's utterance to understand their intent and interests"""
    utterance_lower = utterance.lower()
    
    # Question detection
    if any(word in utterance_lower for word in ['what', 'how', 'why', 'when', 'where', 'who', 'which']) or '?' in utterance:
        if any(word in utterance_lower for word in ['more', 'tell me', 'explain', 'about']):
            return "Asking for more detailed information - wants deeper explanation"
        elif any(word in utterance_lower for word in ['meaning', 'significance', 'important']):
            return "Asking about meaning/significance - wants cultural/historical context"
        elif any(word in utterance_lower for word in ['made', 'created', 'built', 'constructed']):
            return "Asking about creation/construction - wants process/technique information"
        else:
            return "Asking a specific question - wants direct answer"
    
    # Interest/engagement detection
    elif any(word in utterance_lower for word in ['interesting', 'fascinating', 'amazing', 'beautiful', 'incredible']):
        return "Expressing positive interest - engaged and wants to learn more"
    
    # Confusion/clarification detection
    elif any(word in utterance_lower for word in ['confused', 'understand', 'unclear', 'not sure']):
        return "Expressing confusion - needs clarification or simpler explanation"
    
    # Agreement/acknowledgment
    elif any(word in utterance_lower for word in ['yes', 'ok', 'sure', 'i see', 'understand']):
        return "Acknowledging information - ready for next topic or deeper detail"
    
    # Personal connection
    elif any(word in utterance_lower for word in ['reminds me', 'similar', 'like', 'seen']):
        return "Making personal connections - engage with their experience"
    
    else:
        return "General engagement - continue educational dialogue"


# ===== EXPLAIN OPTION FUNCTIONS =====

def build_explain_new_fact_prompt(ex_id: Optional[str], context_section: str,
                                facts_all: List[str], facts_used: List[str],
                                selected_fact: Optional[str], current_completion: float = 0.0) -> str:
    """Build prompt for explaining a new fact about current exhibit"""

    if not facts_all:
        return f"""Museum guide. Current exhibit: {ex_id} (completion: {current_completion:.1%}).

{context_section}

No new facts available. Ask if they'd like to explore a different aspect or move to another exhibit.

Response (1-2 sentences):"""

    facts_list = "\n".join([f"  {fact}" for fact in facts_all])  # Show top 3 facts

    return f"""Museum guide. Current exhibit: {ex_id} (completion: {current_completion:.1%}).

{context_section}

AVAILABLE FACTS (select 1 fact that best matches their interest):
{facts_list}

RULES:
- Choose 1 fact that directly addresses what they just said or asked
- Include the [ID] in brackets after mentioning the fact
- Keep response natural and conversational (2-3 sentences)
- Build on their specific interest or question

Response:"""


def build_repeat_fact_prompt(ex_id: Optional[str], context_section: str,
                           facts_all: List[str], facts_used: List[str],
                           selected_fact: Optional[str], current_completion: float = 0.0) -> str:
    """Build prompt for repeating a previously shared fact"""

    if facts_used:
        fact_to_repeat = selected_fact if selected_fact else facts_used[-1]

        return f"""Museum guide. Current exhibit: {ex_id} (completion: {current_completion:.1%}).

{context_section}

YOUR TASK:
1. Directly address what the visitor just said or asked
2. Rephrase this previously shared fact in NEW WORDS: "{fact_to_repeat}"
3. Keep it brief and conversational (2-3 sentences)
4. Make it more accessible or memorable this time

IMPORTANT: Don't introduce NEW facts - just rephrase what was already shared.
Be natural and responsive to their specific question or confusion!"""
    else:
        return f"""Museum guide. Current exhibit: {ex_id} (completion: {current_completion:.1%}).

{context_section}

No facts shared yet. Share an interesting fact about this exhibit that relates to what they just said.

Response (2-3 sentences):"""


def build_clarify_fact_prompt(ex_id: Optional[str], context_section: str,
                            facts_all: List[str], facts_used: List[str],
                            selected_fact: Optional[str], current_completion: float = 0.0) -> str:
    """Build prompt for clarifying a fact"""
    if facts_used:
        fact_to_clarify = selected_fact if selected_fact else facts_used[-1]
        return f"""Museum guide. Current exhibit: {ex_id} (completion: {current_completion:.1%}).

{context_section}

YOUR TASK:
1. Directly address what the visitor just asked or said
2. Clarify this previously discussed fact using a simple analogy or example: "{fact_to_clarify}"
3. Keep it conversational and brief (2-3 sentences)
4. Use everyday language to make it clearer

IMPORTANT: Don't introduce NEW information - just explain the existing fact more clearly.
Make your response natural and directly address their specific confusion!"""
    else:
        return f"""Museum guide. Current exhibit: {ex_id} (completion: {current_completion:.1%}).

{context_section}

No facts shared yet. Clarify an interesting fact about this exhibit that relates to what they just said.

Response (2-3 sentences):"""


# ===== ASK QUESTION OPTION FUNCTIONS =====

def build_ask_opinion_prompt(ex_id: Optional[str], context_section: str,
                               facts_all: List[str], facts_used: List[str],
                               current_completion: float = 0.0) -> str:
    """Build prompt for asking the visitor's opinion"""

    return f"""Museum guide. Current exhibit: {ex_id} (completion: {current_completion:.1%}).

{context_section}

YOUR TASK:
1. Briefly acknowledge what they just said
2. Ask for their opinion or feeling about the exhibit or what we discussed
3. Keep it natural and conversational (1-2 sentences)

CRITICAL RULES:
- NO role labels or meta-commentary
- NO facts - just respond and ask their view
- Speak naturally and directly
- Make it relevant to what they showed interest in

Response:"""


def build_ask_memory_prompt(ex_id: Optional[str], context_section: str,
                          facts_all: List[str], facts_used: List[str],
                          current_completion: float = 0.0) -> str:
    """Build prompt for checking the visitor's memory"""

    return f"""Museum guide. Current exhibit: {ex_id} (completion: {current_completion:.1%}).

{context_section}

YOUR TASK:
- Answer their question briefly if they asked one
- Ask if they remember something specific we discussed earlier
- Keep it natural and conversational (1-2 sentences)

CRITICAL RULES:
- NO [FACT_ID] tags or repeating facts
- Make it relevant to what they've shown interest in
- Speak naturally and directly

Response:"""


def build_ask_clarification_prompt(ex_id: Optional[str], context_section: str,
                                 facts_all: List[str], facts_used: List[str],
                                 current_completion: float = 0.0) -> str:
    """Build prompt for asking for clarification"""
    return f"""Museum guide. Current exhibit: {ex_id} (completion: {current_completion:.1%}).

{context_section}

YOUR TASK:
- Acknowledge what they said
- Ask what specific aspect interests them most
- Keep it natural and conversational (1-2 sentences)

CRITICAL RULES:
- NO [FACT_ID] tags or sharing facts
- Make it relevant to what they showed interest in
- Speak naturally and directly

Response:"""


# ===== OFFER TRANSITION OPTION FUNCTIONS =====

def _build_exhibit_inventory_section(exhibit_names: List[str], facts_used: List[str], knowledge_graph) -> str:
    """Build a section showing all exhibits and their exploration status"""
    if not exhibit_names or not knowledge_graph:
        return ""
    
    inventory_lines = ["MUSEUM EXHIBITS INVENTORY:", ""]
    
    # Convert facts_used to a set for faster lookup (plain text without IDs)
    facts_used_set = set(facts_used)
    
    # Calculate facts per exhibit
    exhibit_facts_count = {}
    exhibit_facts_used = {}
    
    for exhibit_name in exhibit_names:
        facts_all = knowledge_graph.get_exhibit_facts(exhibit_name) if knowledge_graph else []
        # Strip IDs from facts_all for comparison
        facts_remaining = [f for f in facts_all if knowledge_graph.strip_fact_id(f) not in facts_used_set]
        
        exhibit_facts_count[exhibit_name] = len(facts_all)
        exhibit_facts_used[exhibit_name] = len(facts_all) - len(facts_remaining)
    
    # Sort by unexplored facts (most first)
    sorted_exhibits = sorted(
        exhibit_names,
        key=lambda ex: (exhibit_facts_count.get(ex, 0) - exhibit_facts_used.get(ex, 0)),
        reverse=True
    )
    
    for exhibit_name in sorted_exhibits:
        total = exhibit_facts_count.get(exhibit_name, 0)
        used = exhibit_facts_used.get(exhibit_name, 0)
        remaining = total - used
        
        status_icon = "✓" if used == total else "◐" if used > 0 else "○"
        inventory_lines.append(
            f"  {status_icon} {exhibit_name.replace('_', ' ')}: "
            f"{used}/{total} facts discussed ({remaining} unexplored)"
        )
    
    inventory_lines.append("")
    return "\n".join(inventory_lines)


def build_offer_transition_prompt(ex_id: Optional[str], context_section: str,
                                facts_all: List[str], facts_used: List[str],
                                exhibit_names: List[str] = None, knowledge_graph = None,
                                target_exhibit: str = None, coverage_dict: dict = None) -> str:
    """
    Build prompt for transitioning to another exhibit.

    Uses exhibit completion tracking to choose the best target exhibit.
    - target_exhibit: The exhibit we want to guide visitor to (from env selection logic)
    - coverage_dict: Museum-wide completion stats (from state tracking)
    """

    # Fallback for when we don't have proper state data
    if not target_exhibit:
        return f"""You are a museum guide. SUGGEST moving to a different exhibit.

{context_section}

Respond naturally:
- Suggest visiting another exhibit
- Be conversational and helpful
- Keep it brief (2 sentences)"""

    # Main transition logic using state-driven exhibit selection
    target_name = target_exhibit.replace('_', ' ')
    current_name = ex_id.replace('_', ' ') if ex_id else 'current exhibit'

    # Use exhibit completion data to inform the transition
    if coverage_dict:
        current_stats = coverage_dict.get(ex_id, {"mentioned": 0, "total": 1, "coverage": 0})
        target_stats = coverage_dict.get(target_exhibit, {"mentioned": 0, "total": 1, "coverage": 0})

        return f"""MUSEUM GUIDE - SMART TRANSITION

CURRENT: {current_name} ({current_stats["mentioned"]}/{current_stats["total"]} facts covered)
TARGET: {target_name} ({target_stats["mentioned"]}/{target_stats["total"]} facts covered)

{context_section}

TASK: Suggest moving to {target_name}
- Acknowledge current exhibit briefly
- Make {target_name} sound appealing
- Keep it natural (2 sentences)

Example: "We've covered a lot here. Let's explore the {target_name} next - it has fascinating pieces from the same era."

Response:"""

    # Simple fallback without completion data
    return f"""MUSEUM GUIDE - SUGGEST MOVE

LOCATION: {current_name}
TARGET: {target_name}

{context_section}

TASK: Suggest moving to {target_name}
- Keep it natural (2 sentences)
- Make it enticing
- Mention "{target_name}" explicitly

Response:"""


def build_link_to_other_exhibit_prompt(ex_id: Optional[str], context_section: str,
                                     facts_all: List[str], facts_used: List[str],
                                     exhibit_names: List[str] = None, knowledge_graph = None,
                                     target_exhibit: str = None, coverage_dict: dict = None,
                                     current_completion: float = 0.0) -> str:
    """Build prompt for linking current exhibit to another exhibit thematically"""

    if not target_exhibit:
        return f"""Museum guide. Current exhibit: {ex_id} (completion: {current_completion:.1%}).

{context_section}

CONNECT this exhibit to others in the museum that share similar themes or historical context.

Response (2-3 sentences):"""

    target_name = target_exhibit.replace('_', ' ')
    current_name = ex_id.replace('_', ' ') if ex_id else 'current exhibit'

    return f"""Museum guide. Current exhibit: {current_name} (completion: {current_completion:.1%}).

{context_section}

YOUR TASK:
- Connect {current_name} to {target_name} thematically
- Explain the artistic/historical/cultural connection
- Keep it natural and informative (2-3 sentences)
- Show how they complement each other

Response:"""

# ===== CONCLUDE OPTION FUNCTIONS =====

def build_wrap_up_prompt(ex_id: Optional[str], context_section: str,
                        facts_all: List[str], facts_used: List[str],
                        current_completion: float = 0.0) -> str:
    """Build prompt for wrapping up the visit"""
    return f"""Museum guide. Current exhibit: {ex_id} (completion: {current_completion:.1%}).

{context_section}

YOUR TASK:
- Thank them warmly for visiting
- Express hope they enjoyed the experience
- Keep it natural and conversational (2 sentences)

CRITICAL RULES:
- NO [FACT_ID] tags or recapping information
- Focus on their overall experience
- End on a positive, welcoming note

Response:"""


def build_summarize_key_points_prompt(ex_id: Optional[str], context_section: str, 
                                    facts_all: List[str], facts_used: List[str]) -> str:
    """Build prompt for summarizing key points"""
    if facts_used:
        key_points = facts_used[-3:] if len(facts_used) >= 3 else facts_used
        summary_points = "\n".join([f"- {fact}" for fact in key_points])
        
        return f"""SUMMARIZE these key points briefly: 
{summary_points}

{context_section}

INSTRUCTIONS:
1. Recap 2-3 main points
2. Keep very brief (2-3 sentences max)

CRITICAL RULES:
- NO new facts - only summarize what's been discussed
- NO [FACT_ID] tags
- Focus on the most interesting or important points

Response:"""
    else:
        return f"""Museum guide. Current exhibit: {ex_id} (completion: {current_completion:.1%}).

{context_section}

YOUR TASK:
- Provide a warm conclusion to the visit
- Thank the visitor for their time and engagement
- Express appreciation for their interest
- End on a positive, welcoming note

Response (2 sentences):"""


