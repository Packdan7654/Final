"""
Hierarchical Reinforcement Learning Training Loop

This module implements the main training loop for the HRL museum dialogue agent.
It orchestrates the interaction between the agent, environment, and simulator,
managing episodes, logging, and training statistics.

Key Components:
- Episode Management: Multi-episode training with proper reset and logging
- Agent-Environment Interaction: Step-by-step dialogue simulation
- Simulator Integration: User behavior simulation and state updates
- Training Monitoring: Real-time logging and statistics tracking
- Log Management: JSON serialization and file saving
"""

import random
import numpy as np
import time
import os
from typing import Dict, Any, Optional, List
from src.environment.env import MuseumDialogueEnv
from src.simulator.sim8_adapter import Sim8Simulator
from src.visualization.training_monitor import TrainingMonitor
from src.agent.actor_critic_agent import ActorCriticAgent
from src.training.actor_critic_trainer import ActorCriticTrainer
import json
from datetime import datetime
import torch
import sys
import io

# Fix Windows console encoding for emojis
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


class ExhibitTracker:
    """Tracks agent and user navigation through exhibits and AOIs"""
    
    def __init__(self):
        self.agent_current_exhibit = None
        self.agent_current_aoi = None
        self.user_current_exhibit = None
        self.user_current_aoi = None
        
        self.agent_exhibit_history = []
        self.user_exhibit_history = []
        self.agent_aoi_history = []
        self.user_aoi_history = []
        
        self.exhibits_visited = set()
        self.aois_visited = set()
    
    def update_agent_location(self, exhibit: str, aoi: Optional[str] = None):
        """Track where agent is directing attention"""
        # Check if exhibit changed
        if exhibit != self.agent_current_exhibit:
            self.agent_exhibit_history.append({
                'exhibit': exhibit,
                'turn': len(self.agent_exhibit_history)
            })
            self.exhibits_visited.add(exhibit)
        
        if aoi and aoi != self.agent_current_aoi:
            self.agent_aoi_history.append({
                'aoi': aoi,
                'exhibit': exhibit,
                'turn': len(self.agent_aoi_history)
            })
            self.aois_visited.add(aoi)
        
        self.agent_current_exhibit = exhibit
        if aoi:
            self.agent_current_aoi = aoi
    
    def update_user_location(self, exhibit: str, aoi: str):
        """Track where user is looking"""
        # Check if exhibit changed
        if exhibit != self.user_current_exhibit:
            self.user_exhibit_history.append({
                'exhibit': exhibit,
                'turn': len(self.user_exhibit_history)
            })
        
        if aoi != self.user_current_aoi:
            self.user_aoi_history.append({
                'aoi': aoi,
                'exhibit': exhibit,
                'turn': len(self.user_aoi_history)
            })
        
        self.user_current_exhibit = exhibit
        self.user_current_aoi = aoi
    
    def get_alignment_status(self) -> str:
        """Check if agent and user are aligned on same exhibit/AOI"""
        if self.agent_current_exhibit == self.user_current_exhibit:
            if self.agent_current_aoi == self.user_current_aoi:
                return "ALIGNED"  # Same exhibit and AOI
            else:
                return "SAME_EXHIBIT"  # Same exhibit, different AOI
        else:
            return "MISALIGNED"  # Different exhibits
    
    def reset(self):
        """Reset tracker for new episode"""
        self.__init__()


class HRLTrainingLoop:
    """
    Hierarchical Reinforcement Learning Training Loop
    
    Manages the complete training process for the HRL museum dialogue agent:
    - Episode orchestration and termination
    - Agent-environment-simulator interaction
    - Training statistics and logging
    - Session management and cleanup
    """
    
    def __init__(self, max_episodes: int = 10, max_turns_per_episode: int = 20,
                 knowledge_graph_path: str = None, turn_delay: float = 1.5,
                 learning_rate: float = 3e-4, gamma: float = 0.99,
                 use_actor_critic: bool = True, device: str = 'cpu',
                 show_prompts: bool = False, force_option: str = None,
                 force_subaction: str = None):
        # ===== TRAINING CONFIGURATION =====
        self.max_episodes = max_episodes
        self.max_turns_per_episode = max_turns_per_episode
        self.turn_delay = turn_delay
        self.use_actor_critic = use_actor_critic
        self.device = device
        self._show_prompts = show_prompts
        
        # ===== TESTING/DEBUGGING OPTIONS =====
        self.force_option = force_option
        self.force_subaction = force_subaction
        if self.force_option or self.force_subaction:
            print(f"\n⚠️  TESTING MODE ENABLED")
            if self.force_option:
                print(f"   Force Option: {self.force_option}")
            if self.force_subaction:
                print(f"   Force Subaction: {self.force_subaction}\n")
        
        # ===== COMPONENT INITIALIZATION =====
        # Initialize environment with simplified knowledge graph
        self.env = MuseumDialogueEnv(
            knowledge_graph_path=knowledge_graph_path
        )
        
        # Initialize user simulator (sim8 adapter) with environment exhibits
        self.simulator = Sim8Simulator(exhibits=self.env.exhibit_keys)
        
        # Initialize training monitor for logging and statistics
        self.monitor = TrainingMonitor()
        
        # Initialize LLM timing trackers
        self.agent_llm_times = []
        self.simulator_llm_times = []
        
        # Initialize exhibit/AOI tracker for navigation visualization
        
        # ===== ACTOR-CRITIC AGENT =====
        if self.use_actor_critic:
            # Get state dimension from environment
            state_dim = self.env.observation_space.shape[0]
            
            # Initialize Actor-Critic agent
            self.agent = ActorCriticAgent(
                state_dim=state_dim,
                options=self.env.options,
                subactions=self.env.subactions,
                hidden_dim=256,
                lstm_hidden_dim=128,
                use_lstm=True,
                device=self.device
            )
            
            # Initialize Actor-Critic trainer
            self.trainer = ActorCriticTrainer(
                agent=self.agent,
                learning_rate=learning_rate,
                gamma=gamma,
                device=self.device
            )
            
            # Episode buffer for training
            self.episode_buffer = {
                'states': [],
                'options': [],
                'subactions': [],
                'rewards': [],
                'next_states': [],
                'dones': []
            }
        
        # ===== TRAINING STATE =====
        self.current_episode = 0
        self.total_episodes = 0
        self.total_turns = 0
        self.episode_rewards = []
        
    def run_training(self):
        """Execute the complete training loop"""
        print("=" * 80)
        print(" " * 20 + "HRL MUSEUM DIALOGUE AGENT TRAINING" + " " * 20)
        print("=" * 80)
        print(f"   📊 Configuration:")
        print(f"      • Max Episodes: {self.max_episodes}")
        print(f"      • Max Turns per Episode: {self.max_turns_per_episode}")
        print(f"      • Available Exhibits: {len(self.env.exhibit_keys)} ({', '.join(self.env.exhibit_keys)})")
        print(f"      • Simulator: Persona-based with LLM")
        print("═" * 82)
        
        try:
            # Run training episodes
            for episode in range(self.max_episodes):
                self.current_episode = episode + 1
                print(f"\n📚 Episode {self.current_episode}/{self.max_episodes}")
                print("-" * 40)
                
                # Run single episode
                episode_reward = self._run_episode()
                self.episode_rewards.append(episode_reward)
                
                # Print episode summary
                self._print_episode_summary(episode_reward)
                
                # Check for early termination
                if self._should_terminate_early():
                    print("🛑 Early termination condition met")
                    break
            
            # Finalize training
            self._finalize_training()
            
        except KeyboardInterrupt:
            print("\n⚠️  Training interrupted by user")
            self._finalize_training()
        except Exception as e:
            print(f"\n❌ Training failed with error: {e}")
            import traceback
            traceback.print_exc()
            self._finalize_training()

    def _run_episode(self) -> float:
        """Run a single training episode"""
        # Initialize episode
        obs, info = self.env.reset()
        self.simulator.initialize_session(persona="Agreeable")
        
        # Reset tracker for new episode
        
        # Reset agent if using Actor-Critic
        if self.use_actor_critic:
            self.agent.reset()
            self.episode_buffer = {
                'states': [],
                'options': [],
                'subactions': [],
                'rewards': [],
                'next_states': [],
                'dones': []
            }
        
        episode_reward = 0.0
        turn_count = 0
        
        print("=" * 80)
        print(" " * 25 + "EPISODE INITIALIZATION" + " " * 25)
        print("=" * 80)
        print(f"   🎭 Simulator Persona: Agreeable")
        print(f"   👁️  Starting AOI: {self.simulator.get_current_aoi()}")
        print(f"   🎯 Episode Goal: Explore exhibits and engage visitor")
        print(f"   📋 Available Exhibits: {', '.join(self.env.exhibit_keys)}")
        print("═" * 82)
        print()
        
        # Episode loop
        print(f"   🔄 Starting episode loop (max {self.max_turns_per_episode} turns)...")
        while turn_count < self.max_turns_per_episode:
            turn_count += 1
            
            # Update environment with simulator state
            self._update_environment_state()
            
            # Generate agent action (Actor-Critic or random)
            if self.use_actor_critic:
                action = self._generate_action_actor_critic(obs)
            else:
                action = self._generate_action_random()
            
            # Execute environment step
            next_obs, reward, done, truncated, info = self.env.step(action)
            
            # Track LLM timing
            agent_time = info.get('agent_llm_time', 0.0)
            if agent_time > 0:
                self.agent_llm_times.append(agent_time)
            
            # Store experience in buffer for Actor-Critic training
            if self.use_actor_critic:
                self.episode_buffer['states'].append(obs)
                self.episode_buffer['options'].append(action['option'])
                self.episode_buffer['subactions'].append(action['subaction'])
                self.episode_buffer['rewards'].append(reward)
                self.episode_buffer['next_states'].append(next_obs)
                self.episode_buffer['dones'].append(done)
            
            # Print turn header and agent decision/utterance FIRST
            self._print_turn_header(turn_count, action, info)
            
            # Update simulator with agent response (happens after agent speaks)
            self._update_simulator_state(info)
            
            # Facts already extracted in env.step(), just retrieve from info
            # Add DialogueBERT insights after user state is updated
            self.env.add_dialoguebert_insights_to_info(info)
            
            # Log training step
            simulator_data = self._get_simulator_data()
            self.monitor.update_training_step(
                state=obs,
                action=action,
                reward=reward,
                done=done,
                info=info,
                simulator_data=simulator_data
            )
            
            # Update episode state
            episode_reward += reward
            obs = next_obs
            
            # Print rest of turn status (simulator response, rewards, facts, stats)
            self._print_turn_details(reward, info)
            
            # Check episode termination
            if done:
                print("=" * 80)
                print(" " * 28 + "EPISODE COMPLETED" + " " * 28)
                print("=" * 80)
                print(f"   ✅ Episode finished after {turn_count} turns")
                print(f"   🎯 Final reward: {episode_reward:.3f}")
                print()
                break
        
        # Train Actor-Critic agent on collected experience
        if self.use_actor_critic and len(self.episode_buffer['states']) > 0:
            train_stats = self.trainer.update(
                states=self.episode_buffer['states'],
                options=self.episode_buffer['options'],
                subactions=self.episode_buffer['subactions'],
                rewards=self.episode_buffer['rewards'],
                next_states=self.episode_buffer['next_states'],
                dones=self.episode_buffer['dones']
            )
            
            # Log training statistics
            print(f"   📈 Training Update:")
            print(f"      Policy Loss: {train_stats['policy_loss']:.4f}")
            print(f"      Value Loss: {train_stats['value_loss']:.4f}")
            print(f"      Entropy: {train_stats['entropy']:.4f}")
            print(f"      Mean Advantage: {train_stats['mean_advantage']:.4f}")
            print()
        
        return episode_reward

    def _update_environment_state(self):
        """Update environment with current simulator state"""
        # Get current exhibit from simulator and map to focus index
        current_exhibit = self.simulator.get_current_aoi()
        focus = 0
        if current_exhibit in self.env.exhibit_keys:
            focus = self.env.exhibit_keys.index(current_exhibit) + 1
        
        # Update environment with available simulator data
        self.env.update_user_state(
            focus=focus,
            utterance=""  # No utterance initially; dwell persists until updated by simulator
        )
        
    def _generate_action_actor_critic(self, obs: np.ndarray) -> Dict[str, Any]:
        """Generate agent action using Actor-Critic policy"""
        # ===== TESTING MODE: Force specific actions =====
        if self.force_option or self.force_subaction:
            available_options = self.env._get_available_options()
            
            # Determine which option to use
            if self.force_option:
                if self.force_option not in available_options:
                    print(f"⚠️  Warning: Force option '{self.force_option}' not available. "
                          f"Available: {available_options}")
                    option_idx = available_options.index(available_options[0])
                    option = available_options[0]
                else:
                    option = self.force_option
                    option_idx = available_options.index(self.force_option)
            else:
                option = available_options[0]
                option_idx = 0
            
            # Determine which subaction to use
            available_subactions = self.env._get_available_subactions(option)
            if self.force_subaction:
                if self.force_subaction not in available_subactions:
                    print(f"⚠️  Warning: Force subaction '{self.force_subaction}' not available for '{option}'. "
                          f"Available: {available_subactions}")
                    subaction_idx = 0
                    subaction = available_subactions[0]
                else:
                    subaction = self.force_subaction
                    subaction_idx = available_subactions.index(self.force_subaction)
            else:
                subaction_idx = 0
                subaction = available_subactions[0]
            
            print(f"🎯 FORCED ACTION: Option={option}, Subaction={subaction}")
            
            return {
                "option": option_idx,
                "subaction": subaction_idx,
                "terminate_option": False
            }
        
        # ===== NORMAL MODE: Use Actor-Critic policy =====
        # Get available options and subactions
        available_options = self.env._get_available_options()
        if not available_options:
            return {"option": 0, "subaction": 0, "terminate_option": False}
        
        # Build available subactions dict
        available_subactions_dict = {}
        for opt in available_options:
            available_subactions_dict[opt] = self.env._get_available_subactions(opt)
        
        # Select action using Actor-Critic agent
        action_info = self.agent.select_action(
            state=obs,
            available_options=available_options,
            available_subactions_dict=available_subactions_dict,
            deterministic=False  # Use stochastic policy during training
        )
        
        return {
            "option": action_info['option'],
            "subaction": action_info['subaction'],
            "terminate_option": action_info['terminated']
        }
    
    def _generate_action_random(self) -> Dict[str, Any]:
        """Generate random action (baseline/fallback)"""
        # Get available options and subactions
        available_options = self.env._get_available_options()
        if not available_options:
            return {"option": 0, "subaction": 0, "terminate_option": False}
        
        # Select random option
        option = random.choice(available_options)
        option_idx = self.env.options.index(option)
        
        # Get available subactions for selected option
        available_subactions = self.env._get_available_subactions(option)
        if not available_subactions:
            return {"option": option_idx, "subaction": 0, "terminate_option": False}
        
        # Select random subaction
        subaction = random.choice(available_subactions)
        subaction_idx = self.env.subactions[option].index(subaction)
        
        # Random termination (10% chance)
        terminate_option = random.random() < 0.1
        
        return {
            "option": option_idx,
            "subaction": subaction_idx,
            "terminate_option": terminate_option
        }

    def _update_simulator_state(self, info: Dict[str, Any]):
        """Update simulator with agent response"""
        agent_utterance = info.get("agent_utterance", "")
        if agent_utterance:
            # Generate user response
            user_response = self.simulator.generate_user_response(agent_utterance)
            
            # Track simulator LLM timing
            sim_time = user_response.get('simulator_llm_time', 0.0)
            if sim_time > 0:
                self.simulator_llm_times.append(sim_time)

            # Update environment with utterance
            if user_response.get("utterance") is not None:
                self.env.update_user_state(utterance=user_response["utterance"])

            # Dwell time reward: take first gaze feature as dwell
            gaze_feats = user_response.get("gaze_features") or []
            if gaze_feats:
                dwell_time = float(gaze_feats[0])
                self.env.update_user_state(dwell=dwell_time)
            
            # Update focus based on simulator's new exhibit (if changed)
            new_exhibit = self.simulator.get_current_aoi()
            new_aoi = user_response.get("aoi", new_exhibit)  # Get AOI from response
            if new_exhibit in self.env.exhibit_keys:
                new_focus = self.env.exhibit_keys.index(new_exhibit) + 1
                self.env.update_user_state(focus=new_focus)

    def _get_simulator_data(self) -> Dict[str, Any]:
        """Get current simulator data for logging"""
        return {
            "aoi": self.simulator.get_current_aoi(),
            "persona": getattr(self.simulator, 'current_persona', 'Unknown'),
            "utterance": "",
            "dwell_time": 0.5,
            "engagement_level": 0.5
        }

    def _print_turn_header(self, turn: int, action: Dict[str, Any], info: Dict[str, Any]):
        """Print turn header, agent decision, and agent utterance (happens BEFORE simulator response)"""
        option = info.get("option", "Unknown")
        subaction = info.get("subaction", "Unknown")
        agent_utterance = info.get("agent_utterance", "")
        terminated_option = info.get("terminated_option", False)
        current_option = info.get("current_option", None)
        turns_in_option = info.get("turns_in_option", 0)
        
        # Determine option status
        # Check if this is truly a new option or just continuing with different subaction
        if hasattr(self, '_previous_option_name'):
            is_same_option = (self._previous_option_name == option)
        else:
            is_same_option = (current_option == option)
            
        option_status = "🔄 CONTINUING" if is_same_option else "🆕 NEW"
        termination_text = " | ⛔ TERMINATED" if terminated_option else ""
        
        # Store for next turn comparison
        if terminated_option:
            # Store the option name even though it terminated
            self._previous_option_name = option
        else:
            self._previous_option_name = option
        
        # Print turn header
        print("-" * 80)
        print(f" TURN {turn:2d} " + "-" * 66)
        print("-" * 80)
        
        # AGENT DECISION SECTION
        print("🤖 AGENT DECISION:")
        print(f"   📋 Option: {option} ({option_status}) | Turns in option: {turns_in_option}{termination_text}")
        print(f"   ⚡ Subaction: {subaction}")
        print(f"   🎲 Raw Action: option={action.get('option', '?')}, subaction={action.get('subaction', '?')}, terminate={action.get('terminate_option', '?')}")
        print()
        
        # AGENT UTTERANCE SECTION
        print("💬 AGENT UTTERANCE:")
        if agent_utterance:
            # Word wrap the utterance nicely
            words = agent_utterance.split()
            lines = []
            current_line = "   \"" 
            for word in words:
                if len(current_line + word + " ") > 75:
                    lines.append(current_line)
                    current_line = "   " + word + " "
                else:
                    current_line += word + " "
            lines.append(current_line.rstrip() + "\"")
            
            for line in lines:
                print(line)
        else:
            print("   \"[No utterance generated]\"")
        print()
        
        # Show LLM prompt (optional, can be toggled)
        llm_prompt = info.get("llm_prompt", "")
        if llm_prompt and hasattr(self, '_show_prompts') and self._show_prompts:
            print("📋 LLM PROMPT:")
            prompt_lines = llm_prompt.split('\n')
            for line in prompt_lines[:15]:  # Show first 15 lines
                if line.strip():
                    print(f"   {line[:75]}")
            if len(prompt_lines) > 15:
                print(f"   ... ({len(prompt_lines) - 15} more lines)")
            print()
    
    def _print_turn_details(self, reward: float, info: Dict[str, Any]):
        """Print simulator response, rewards, facts, and stats (happens AFTER simulator responds)"""
        current_focus = info.get("current_focus", 0)
        current_exhibit = info.get("current_exhibit", "Unknown")
        facts_shared = info.get("facts_shared", 0)
        exhibits_covered = info.get("exhibits_covered", 0)
        
        # Get simulator data
        simulator_state = self.simulator.get_current_state()
        user_response = simulator_state.get("last_user_response", {})
        # SIMULATOR RESPONSE SECTION
        print("🧠 SIMULATOR RESPONSE:")
        user_utterance = user_response.get("utterance")
        user_aoi = user_response.get("aoi", "Unknown")
        user_persona = user_response.get("persona", "Unknown")
        response_type = user_response.get("response_type", "unknown")
        gaze_features = user_response.get("gaze_features", [])
        
        # Get visitor's current exhibit from simulator
        sim_state = self.simulator.get_current_state()
        visitor_exhibit = sim_state.get("current_exhibit", "Unknown")
        
        if user_utterance:
            print(f"   👤 User Says: \"{user_utterance}\"")
        else:
            print(f"   👤 User Says: [SILENT] ({response_type})")
        
        print(f"   🏛️  Visitor at Exhibit: {visitor_exhibit} | 👁️  Looking at AOI: {user_aoi}")
        print(f"   🎭 Persona: {user_persona} | Response Type: {response_type}")
        
        # Gaze features display
        if gaze_features and len(gaze_features) >= 6:
            dwell = gaze_features[0]
            saccade = gaze_features[1] 
            entropy = gaze_features[2]
            fix_rate = gaze_features[3]
            dom_ratio = gaze_features[4]
            entry_lat = gaze_features[5]
            
            print(f"   📊 Gaze: Dwell={dwell:.3f} | Saccade={saccade:.3f} | Entropy={entropy:.3f}")
            print(f"           FixRate={fix_rate:.3f} | DomRatio={dom_ratio:.3f} | EntryLat={entry_lat:.3f}")
        else:
            print(f"   📊 Gaze: [No gaze data available]")
        print()
        
        # REWARD BREAKDOWN SECTION
        print("💰 REWARD BREAKDOWN:")
        engagement_reward = info.get("reward_engagement", 0.0)
        novelty_reward = info.get("reward_novelty", 0.0)
        
        print(f"   🔥 Engagement:    {engagement_reward:+7.3f} (lagged from prev turn)")
        print(f"   ✨ Novelty:       {novelty_reward:+7.3f} (verified facts)")
        print(f"   📈 TOTAL:         {reward:+7.3f}")
        print()
        
        # FACT IDs EXTRACTED
        mentioned_fact_ids = info.get("facts_mentioned_in_utterance", [])
        hallucinated_fact_ids = info.get("hallucinated_facts", [])
        
        if mentioned_fact_ids or hallucinated_fact_ids:
            if mentioned_fact_ids:
                print(f"📝 NEW FACTS: {len(mentioned_fact_ids)} fact(s) - {mentioned_fact_ids}")
            if hallucinated_fact_ids:
                print(f"❌ HALLUCINATED: {len(hallucinated_fact_ids)} fact(s) - {hallucinated_fact_ids} (no reward)")
            print()
        
        # DialogueBERT INSIGHTS SECTION
        dialoguebert = info.get("dialoguebert_insights")
        if dialoguebert:
            print("🧩 DialogueBERT INSIGHTS:")
            print(f"   🏷️ Intent: {dialoguebert.get('intent_category', 'unknown')}")
            print(f"   📏 ‖intent‖={dialoguebert.get('intent_norm', 0.0):.3f} | ‖context‖={dialoguebert.get('context_norm', 0.0):.3f}")
            print(f"   🔗 cos(intent, context)={dialoguebert.get('cosine_intent_context', 0.0):+.3f}")
            print(f"   🔄 cos(intent, prev_intent)={dialoguebert.get('cosine_intent_prev', 0.0):+.3f} | cos(context, prev_context)={dialoguebert.get('cosine_context_prev', 0.0):+.3f}")
            print()
        
        # TRAINING STATS SECTION
        print("📈 TRAINING STATS:")
        print(f"   🏛️  Current Exhibit: {current_exhibit} | 👁️  Focus Index: {current_focus}")
        print(f"   📚 Facts Shared: {facts_shared} | 🎨 Exhibits Covered: {exhibits_covered}")
        
        # Show available actions for next turn
        available_options = info.get("available_options", [])
        available_subactions = info.get("available_subactions", [])
        print(f"   🎮 Next Available - Options: {available_options}")
        print(f"                      Subactions: {available_subactions}")
        
        print("─" * 82)
        print()  # Add spacing between turns
        
        # Small delay to make live viewing easier
        time.sleep(self.turn_delay)

    def _print_episode_summary(self, episode_reward: float):
        """Print detailed episode summary"""
        if not self.monitor.training_history:
            return
            
        last_turn = self.monitor.training_history[-1]
        info = last_turn['info']
        
        # Calculate episode statistics
        episode_turns = [step for step in self.monitor.training_history if step['episode'] == self.current_episode]
        total_turns = len(episode_turns)
        avg_reward_per_turn = episode_reward / total_turns if total_turns > 0 else 0.0
        
        # Count action types used in this episode
        action_counts = {}
        for turn in episode_turns:
            option = turn['info'].get('option', 'Unknown')
            action_counts[option] = action_counts.get(option, 0) + 1
        
        print(f"   📊 Episode {self.current_episode} Summary:")
        print(f"      🎯 Performance: Total Reward={episode_reward:.3f} | Avg/Turn={avg_reward_per_turn:.3f} | Turns={total_turns}")
        print(f"      📚 Content: Facts Shared={info.get('facts_shared', 0)} | Exhibits Covered={info.get('exhibits_covered', 0)}")
        print(f"      🎭 Final State: Focus={info.get('current_exhibit', 'Unknown')} | Dwell={info.get('dwell', 0.0):.2f}")
        
        # Show action distribution
        if action_counts:
            action_str = " | ".join([f"{opt}={count}" for opt, count in action_counts.items()])
            print(f"      🎮 Actions: {action_str}")
        
        # Print detailed exhibit/facts progress
        self._print_exhibit_facts_progress()
        
        print()  # Add spacing between episodes

    def _print_exhibit_facts_progress(self):
        """SIMPLE: Print which facts were mentioned per exhibit"""
        print()
        print("      " + "=" * 100)
        print(f"      📍 EXHIBIT PROGRESS [Episode {self.current_episode}]")
        print("      " + "=" * 100)
        
        total_mentioned = 0
        total_facts = 0
        
        for exhibit in sorted(self.env.exhibit_keys):
            all_facts = self.env.knowledge_graph.get_exhibit_facts(exhibit)
            mentioned_ids = self.env.facts_mentioned_per_exhibit[exhibit]
            
            total_facts += len(all_facts)
            total_mentioned += len(mentioned_ids)
            
            print(f"\n      📌 {exhibit} ({len(mentioned_ids)}/{len(all_facts)} facts)")
            print(f"      {'-' * 96}")
            
            for fact in all_facts:
                fact_id = self.env.knowledge_graph.extract_fact_id(fact)
                fact_text = self.env.knowledge_graph.strip_fact_id(fact)
                if fact_id in mentioned_ids:
                    print(f"      ✓ {fact_id}  [mentioned]     {fact_text[:70]}")
                else:
                    print(f"      ○ {fact_id}  [not mentioned] {fact_text[:70]}")
        
        print("      " + "=" * 100)
        pct = (total_mentioned / total_facts * 100) if total_facts > 0 else 0
        print(f"      📊 TOTAL: {total_mentioned}/{total_facts} facts ({pct:.1f}%)")
        print("      " + "=" * 100)
        print()

    def _should_terminate_early(self) -> bool:
        """Check if training should terminate early"""
        # Terminate if average reward is consistently high
        if len(self.episode_rewards) >= 3:
            recent_avg = np.mean(self.episode_rewards[-3:])
            if recent_avg > 15.0:  # High performance threshold
                return True
        
        return False

    def _finalize_training(self):
        """Finalize training and save results"""
        print("\n" + "=" * 60)
        print("🏁 Finalizing Training")
        print("=" * 60)
        
        # Update final statistics
        self.total_episodes = self.current_episode
        self.total_turns = len(self.monitor.training_history)
        
        # Get training summary
        summary = self.monitor.get_training_summary()
        
        # Print final statistics
        print(f"📈 Training Summary:")
        print(f"   Total Episodes: {summary['total_episodes']}")
        print(f"   Total Turns: {summary['total_turns']}")
        print(f"   Average Reward per Turn: {summary['average_reward_per_turn']:.4f}")
        print(f"   Average Reward per Episode: {summary['average_reward_per_episode']:.4f}")
        print(f"   Best Episode Reward: {summary['best_episode_reward']:.4f}")
        print(f"   Most Used Option: {summary['most_used_option']}")
        print(f"   Most Used Subaction: {summary['most_used_subaction']}")
        
        # Print LLM timing summary
        if self.agent_llm_times or self.simulator_llm_times:
            print(f"\n⏱️  LLM Timing Summary:")
            if self.agent_llm_times:
                avg_agent = sum(self.agent_llm_times) / len(self.agent_llm_times)
                total_agent = sum(self.agent_llm_times)
                print(f"   🤖 Agent LLM:     {len(self.agent_llm_times)} calls | Avg: {avg_agent:.2f}s | Total: {total_agent:.1f}s")
            if self.simulator_llm_times:
                avg_sim = sum(self.simulator_llm_times) / len(self.simulator_llm_times)
                total_sim = sum(self.simulator_llm_times)
                print(f"   👤 Simulator LLM: {len(self.simulator_llm_times)} calls | Avg: {avg_sim:.2f}s | Total: {total_sim:.1f}s")
            if self.agent_llm_times and self.simulator_llm_times:
                total_all = sum(self.agent_llm_times) + sum(self.simulator_llm_times)
                count_all = len(self.agent_llm_times) + len(self.simulator_llm_times)
                avg_all = total_all / count_all if count_all > 0 else 0
                print(f"   ⚡ Total LLM:     {count_all} calls | Avg: {avg_all:.2f}s | Total: {total_all:.1f}s")
        
        # Save training log
        self._save_training_log()
        
        # Clean up
        self.monitor.close()
        print("✅ Training completed successfully!")

    def _save_training_log(self):
        """Save training results to JSON file in date-organized directory"""
        try:
            # Create date-based directory structure
            now = datetime.now()
            date_folder = now.strftime("%Y-%m-%d")
            timestamp = now.strftime("%H%M%S")
            
            # Create logs directory if it doesn't exist
            logs_dir = "training_logs"
            if not os.path.exists(logs_dir):
                os.makedirs(logs_dir)
            
            # Create date-specific directory
            date_dir = os.path.join(logs_dir, date_folder)
            if not os.path.exists(date_dir):
                os.makedirs(date_dir)
                print(f"📁 Created new log directory: {date_dir}")
            
            # Generate filename with timestamp
            filename = f"training_log_{timestamp}.json"
            filepath = os.path.join(date_dir, filename)
            
            # Prepare data for JSON serialization
            training_data = {
                "training_summary": self.monitor.get_training_summary(),
                "episode_rewards": self.episode_rewards,
                "training_history": []
            }
            
            # Convert training history to JSON-serializable format
            for step in self.monitor.training_history:
                serializable_step = {
                    "episode": step["episode"],
                    "turn": step["turn"],
                    "action": step["action"],
                    "reward": step["reward"],
                    "done": step["done"],
                    "info": {
                        "option": step["info"].get("option"),
                        "subaction": step["info"].get("subaction"),
                        "agent_utterance": step["info"].get("agent_utterance"),
                        "facts_shared": step["info"].get("facts_shared"),
                        "exhibits_covered": step["info"].get("exhibits_covered"),
                        "current_focus": step["info"].get("current_focus"),
                        "current_exhibit": step["info"].get("current_exhibit")
                    }
                }
                
                # Add simulator data if available
                if step.get("simulator_data"):
                    serializable_step["simulator_data"] = {
                        "aoi": step["simulator_data"].get("aoi"),
                        "persona": step["simulator_data"].get("persona"),
                        "utterance": step["simulator_data"].get("utterance")
                    }
                
                training_data["training_history"].append(serializable_step)
            
            # Save to file
            with open(filepath, 'w') as f:
                json.dump(training_data, f, indent=2)
            
            print(f"💾 Training log saved to: {filepath}")
            
        except Exception as e:
            print(f"❌ Error saving training log: {e}")


def main():
    """Main training function"""
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Create and run training loop
    training_loop = HRLTrainingLoop(max_episodes=5, max_turns_per_episode=15)
    training_loop.run_training()


if __name__ == "__main__":
    main()
