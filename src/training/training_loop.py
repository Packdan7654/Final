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
from src.visualization.live_training_monitor import LiveTrainingMonitor
from src.visualization.metrics_tracker import MetricsTracker
from src.visualization.museum_map_visualizer import MuseumMapVisualizer
from src.visualization.live_progress import LiveProgressTracker
from src.agent.actor_critic_agent import ActorCriticAgent
from src.training.actor_critic_trainer import ActorCriticTrainer
from src.utils.detailed_logger import DetailedEpisodeLogger
import json
from datetime import datetime
from pathlib import Path
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
                 force_subaction: str = None, enable_live_monitor: bool = False,
                 save_metrics: bool = True, enable_map_viz: bool = True,
                 save_map_frames: bool = False, live_map_display: bool = False,
                 map_interval: int = 50, verbose: bool = False):
        # ===== TRAINING CONFIGURATION =====
        self.max_episodes = max_episodes
        self.max_turns_per_episode = max_turns_per_episode
        self.turn_delay = turn_delay
        self.use_actor_critic = use_actor_critic
        self.device = device
        self._show_prompts = show_prompts
        self.verbose = verbose  # If False, use clean progress tracker
        
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
        # Load knowledge graph (single source of truth)
        from src.utils.knowledge_graph import SimpleKnowledgeGraph
        self.knowledge_graph = SimpleKnowledgeGraph(knowledge_graph_path)
        
        # Initialize environment with knowledge graph
        self.env = MuseumDialogueEnv(
            knowledge_graph_path=knowledge_graph_path,
            max_turns=max_turns_per_episode
        )
        
        # Initialize user simulator with knowledge graph (NOT just exhibit list)
        self.simulator = Sim8Simulator(knowledge_graph=self.knowledge_graph)
        
        # Initialize training monitor for logging and statistics
        self.monitor = TrainingMonitor()
        
        # ===== LIVE MONITORING SYSTEM =====
        self.enable_live_monitor = enable_live_monitor
        self.save_metrics = save_metrics
        
        # Initialize live training monitor (optional)
        self.live_monitor = LiveTrainingMonitor(
            enabled=enable_live_monitor,
            log_dir="training_logs"
        )
        
        # Initialize metrics tracker (always created, saved conditionally)
        self.metrics_tracker = MetricsTracker()
        
        # Initialize live progress tracker (clean output)
        self.progress_tracker = LiveProgressTracker(max_episodes=max_episodes)
        
        # ===== MUSEUM MAP VISUALIZATION =====
        self.enable_map_viz = enable_map_viz
        self.save_map_frames = save_map_frames
        self.map_interval = map_interval
        
        # Determine map save directory (use experiment dir if available)
        experiment_dir = os.environ.get('EXPERIMENT_DIR', None)
        if experiment_dir:
            map_save_dir = os.path.join(experiment_dir, 'maps')
        else:
            map_save_dir = "training_logs/maps"
        
        # Initialize map visualizer (optional)
        self.map_visualizer = MuseumMapVisualizer(
            enabled=enable_map_viz,
            exhibits=self.env.exhibit_keys,
            save_dir=map_save_dir,
            live_display=False  # Save files but don't display live
        )
        
        # Initialize LLM timing trackers
        self.agent_llm_times = []
        self.simulator_llm_times = []
        
        # Initialize detailed episode logger
        experiment_dir = os.environ.get('EXPERIMENT_DIR', None)
        if experiment_dir:
            self.detailed_logger = DetailedEpisodeLogger(Path(experiment_dir))
        else:
            self.detailed_logger = None
        
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
        # Record start time for overnight session tracking
        self.start_time = time.time()
        
        # Show configuration
        print("=" * 80)
        print("HRL MUSEUM DIALOGUE TRAINING")
        print("=" * 80)
        print(f"Episodes: {self.max_episodes} | Max Turns: {self.max_turns_per_episode} | Device: {self.device}")
        print(f"Exhibits: {len(self.env.exhibit_keys)} | Checkpoints: Every 50 episodes")
        print(f"Started: {datetime.fromtimestamp(self.start_time).strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        if not self.verbose:
            print("\n[Progress Mode: Episode-level tracking. Use --verbose for turn details]\n")
        
        try:
            # Run training episodes
            for episode in range(self.max_episodes):
                self.current_episode = episode + 1
                
                # Start episode tracking
                self.progress_tracker.start_episode(self.current_episode)
                
                if self.verbose:
                    print(f"\n📚 Episode {self.current_episode}/{self.max_episodes}")
                    print("-" * 40)
                
                # Run single episode
                episode_reward, episode_length = self._run_episode()
                self.episode_rewards.append(episode_reward)
                
                # End episode tracking (prints summary)
                self.progress_tracker.end_episode(episode_reward, episode_length)
                
                if self.verbose:
                    # Print verbose episode summary
                    self._print_episode_summary(episode_reward)
                
                # Progress report every 50 episodes for overnight monitoring
                if self.current_episode % 50 == 0:
                    if self.verbose:
                        self._print_progress_report()
                    
                    # Save incremental checkpoint (metrics + model)
                    self._save_checkpoint(f"checkpoint_ep{self.current_episode}")
                
                # Check for early termination
                if self._should_terminate_early():
                    print("\n🛑 Early termination condition met")
                    break
            
            # Finalize training
            self._finalize_training()
            
        except KeyboardInterrupt:
            print("\n⚠️  Training interrupted by user")
            self._finalize_training()
        except Exception as e:
            # Check if it's an LLM critical error
            from src.utils.llm_handler import LLMCriticalError
            if isinstance(e, LLMCriticalError):
                print("\n" + "🚨" * 80)
                print("CRITICAL LLM ERROR - GRACEFUL SHUTDOWN")
                print("🚨" * 80)
                print(f"\nError: {e}")
                print(f"\nTraining stopped at Episode {self.current_episode}/{self.max_episodes}")
                print(f"Saving model and metrics up to this point...\n")
                self._finalize_training()
                print("\n✅ Model and metrics saved successfully!")
                print("Training can resume from this checkpoint later.")
                print("=" * 80)
            else:
                print(f"\n❌ Training failed with error: {e}")
                import traceback
                traceback.print_exc()
                self._finalize_training()

    def _run_episode(self) -> float:
        """Run a single training episode"""
        # Initialize episode
        obs, info = self.env.reset()
        self.simulator.initialize_session(persona="Agreeable")
        
        # Start detailed logging for this episode
        if self.detailed_logger:
            self.detailed_logger.start_episode(self.current_episode)
        
        # Sync initial simulator state to environment
        # This ensures the agent sees the correct starting exhibit focus
        self._update_environment_state()
        obs = self.env._get_obs()  # Get fresh observation with correct focus
        
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
        
        # ===== LIVE MONITOR: Episode Start =====
        self.live_monitor.on_episode_start(
            episode_num=self.current_episode,
            persona=self.simulator.current_persona,
            exhibits=self.env.exhibit_keys
        )
        
        # ===== MAP VISUALIZER: Reset for Episode =====
        self.map_visualizer.reset()
        
        episode_reward = 0.0
        turn_count = 0
        
        if self.verbose:
            print("=" * 80)
            print(" " * 25 + "EPISODE INITIALIZATION" + " " * 25)
            print("=" * 80)
            print(f"   🎭 Simulator Persona: Agreeable")
            print(f"   👁️  Starting AOI: {self.simulator.get_current_aoi()}")
            print(f"   🎯 Episode Goal: Explore exhibits and engage visitor")
            print(f"   📋 Available Exhibits: {', '.join(self.env.exhibit_keys)}")
            print("═" * 82)
            print()
            print(f"   🔄 Starting episode loop (max {self.max_turns_per_episode} turns)...")
        
        # Episode loop
        while turn_count < self.max_turns_per_episode:
            turn_count += 1
            
            # Verbose: Print turn header and current state FIRST (before action selection)
            if self.verbose:
                print("-" * 80)
                print(f" TURN {turn_count:2d} " + "-" * 66)
                print("-" * 80)
                self._print_state_vector(obs)
            
            # Update environment with simulator state
            self._update_environment_state()
            
            # Generate agent action (Actor-Critic or random)
            if self.use_actor_critic:
                action = self._generate_action_actor_critic(obs)
            else:
                action = self._generate_action_random()
            
            # Execute environment step
            next_obs, reward, done, truncated, info = self.env.step(action)
            
            # Update progress tracker with turn info (no printing)
            option = info.get('option', 'Unknown')
            self.progress_tracker.update_turn(reward, option)
            
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
            
            # Verbose: Print agent decision and utterance (after action execution)
            if self.verbose:
                self._print_agent_decision(action, info)
            
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
            
            # ===== LIVE MONITOR & METRICS: Turn Update =====
            # Get simulator data for turn tracking
            simulator_state = self.simulator.get_current_state()
            user_response = simulator_state.get("last_user_response", {})
            
            turn_data = {
                "agent_utterance": info.get("agent_utterance", ""),
                "user_utterance": user_response.get("utterance", ""),
                "option": info.get("option", "Unknown"),
                "subaction": info.get("subaction", "Unknown"),
                "dwell": user_response.get("gaze_features", [0.0])[0],
                "response_type": user_response.get("response_type", "unknown"),
                "current_exhibit": info.get("current_exhibit", "Unknown"),
                "facts_shared": info.get("facts_shared", 0),
                "reward_engagement": info.get("reward_engagement", 0.0),
                "reward_novelty": info.get("reward_novelty", 0.0),
                "reward_responsiveness": info.get("reward_responsiveness", 0.0),
                "reward_transition_penalty": info.get("reward_transition_penalty", 0.0),
                "reward_conclude": info.get("reward_conclude", 0.0),
                "reward_spam_penalty": info.get("reward_spam_penalty", 0.0),
                "total_reward": reward,
                "exhibit_coverage": self.env._get_museum_exhibit_coverage()
            }
            
            # Update live monitor
            self.live_monitor.on_turn(turn_data)
            
            # Update metrics tracker
            self.metrics_tracker.update_turn(turn_data)
            
            # Log detailed episode data
            if self.detailed_logger:
                # Get prompts from environment and simulator
                agent_prompt = getattr(self.env, '_last_llm_prompt', None)
                agent_system_prompt = getattr(self.env, '_last_agent_system_prompt', None)
                simulator_prompt = getattr(self.simulator, '_last_simulator_prompt', None)
                simulator_system_prompt = getattr(self.simulator, '_last_simulator_system_prompt', None)
                
                self.detailed_logger.log_turn(
                    turn_num=turn_count,
                    state=obs,
                    action=action,
                    reward=reward,
                    info=info,
                    next_state=next_obs,
                    agent_prompt=agent_prompt,
                    agent_system_prompt=agent_system_prompt,
                    simulator_prompt=simulator_prompt,
                    simulator_system_prompt=simulator_system_prompt,
                    user_response=user_response
                )
            
            # ===== MAP VISUALIZER: Update Turn =====
            if self.enable_map_viz:
                agent_exhibit = info.get("current_exhibit", "Unknown")
                visitor_exhibit = simulator_state.get("current_exhibit", agent_exhibit)
                current_dwell = user_response.get("gaze_features", [0.0])[0]
                current_option = info.get("option", "Unknown")
                
                # Get completion rates for all exhibits
                coverage = self.env._get_museum_exhibit_coverage()
                exhibit_completion = {
                    exhibit_name: coverage.get(exhibit_name, {"coverage": 0.0})["coverage"]
                    for exhibit_name in self.env.exhibit_keys
                }
                
                self.map_visualizer.update(
                    agent_exhibit=agent_exhibit,
                    visitor_exhibit=visitor_exhibit,
                    dwell=current_dwell,
                    turn_num=turn_count,
                    option=current_option,
                    exhibit_completion=exhibit_completion
                )
                
                # Capture frame for animation
                self.map_visualizer.capture_frame()
                
                # Save snapshot if requested
                if self.save_map_frames:
                    # Save every turn when --save-map-frames is enabled
                    episode_dir = f"episode_{self.current_episode:03d}"
                    self.map_visualizer.save_snapshot(
                        f"{episode_dir}/turn_{turn_count:02d}.png"
                    )
            
            obs = next_obs
            
            # Verbose: Print rest of turn status (simulator response, rewards, facts, stats)
            if self.verbose:
                self._print_turn_details(reward, info)
            
            # Check episode termination
            if done:
                if self.verbose:
                    print("=" * 80)
                    print(" " * 28 + "EPISODE COMPLETED" + " " * 28)
                    print("=" * 80)
                    print(f"   ✅ Episode finished after {turn_count} turns")
                    print(f"   🎯 Final reward: {episode_reward:.3f}")
                    print()
                
                # End detailed logging
                if self.detailed_logger:
                    episode_stats = {
                        "total_turns": turn_count,
                        "episode_reward": episode_reward,
                        "avg_reward_per_turn": episode_reward / turn_count if turn_count > 0 else 0.0
                    }
                    self.detailed_logger.end_episode(episode_reward, episode_stats)
                
                break
        
        # End detailed logging if episode ended without done flag (max turns reached)
        if self.detailed_logger and turn_count >= self.max_turns_per_episode:
            episode_stats = {
                "total_turns": turn_count,
                "episode_reward": episode_reward,
                "avg_reward_per_turn": episode_reward / turn_count if turn_count > 0 else 0.0
            }
            self.detailed_logger.end_episode(episode_reward, episode_stats)
        
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
            if self.verbose:
                print(f"   📈 Training Update:")
                print(f"      Policy Loss: {train_stats['policy_loss']:.4f}")
                print(f"      Value Loss: {train_stats['value_loss']:.4f}")
                print(f"      Entropy: {train_stats['entropy']:.4f}")
                print(f"      Mean Advantage: {train_stats['mean_advantage']:.4f}")
                print()
        
        # ===== LIVE MONITOR & METRICS: Episode End =====
        # Count exhibits covered
        exhibits_covered = sum(1 for fact_set in self.env.facts_mentioned_per_exhibit.values() 
                              if len(fact_set) > 0)
        total_facts = sum(len(fact_set) for fact_set in self.env.facts_mentioned_per_exhibit.values())
        
        # Calculate coverage ratio
        total_available_facts = sum(len(self.env.knowledge_graph.get_exhibit_facts(ex)) 
                                   for ex in self.env.exhibit_keys)
        coverage_ratio = total_facts / total_available_facts if total_available_facts > 0 else 0.0
        
        # Build episode summary
        episode_summary = {
            "cumulative_reward": episode_reward,
            "turns": turn_count,
            "coverage_ratio": coverage_ratio,
            "total_facts": total_facts,
            "exhibits_covered": exhibits_covered,
            "reward_engagement": sum(info.get("reward_engagement", 0.0) for info in 
                                    [self.monitor.training_history[i]['info'] 
                                     for i in range(len(self.monitor.training_history))
                                     if self.monitor.training_history[i].get('episode') == self.current_episode]),
            "reward_novelty": sum(info.get("reward_novelty", 0.0) for info in 
                                 [self.monitor.training_history[i]['info'] 
                                  for i in range(len(self.monitor.training_history))
                                  if self.monitor.training_history[i].get('episode') == self.current_episode]),
            "reward_responsiveness": sum(info.get("reward_responsiveness", 0.0) for info in 
                                        [self.monitor.training_history[i]['info'] 
                                         for i in range(len(self.monitor.training_history))
                                         if self.monitor.training_history[i].get('episode') == self.current_episode]),
            "reward_transition": sum(info.get("reward_transition_penalty", 0.0) for info in 
                                    [self.monitor.training_history[i]['info'] 
                                     for i in range(len(self.monitor.training_history))
                                     if self.monitor.training_history[i].get('episode') == self.current_episode]),
            "reward_conclude": sum(info.get("reward_conclude", 0.0) for info in 
                                  [self.monitor.training_history[i]['info'] 
                                   for i in range(len(self.monitor.training_history))
                                   if self.monitor.training_history[i].get('episode') == self.current_episode])
        }
        
        # Update live monitor
        self.live_monitor.on_episode_end(episode_summary)
        
        # Update metrics tracker
        self.metrics_tracker.update_episode(episode_summary)
        
        # ===== MAP VISUALIZER: Save Episode Animation (at specified interval) =====
        if self.enable_map_viz and (self.current_episode % self.map_interval == 0):
            animation_filename = f"episode_{self.current_episode:03d}_animation.gif"
            self.map_visualizer.save_animation(animation_filename, fps=2)
            print(f"   🗺️ Map animation saved: training_logs/maps/{animation_filename}")
        elif self.enable_map_viz:
            print(f"   🗺️ Map episode {self.current_episode} (animation saved every {self.map_interval} episodes)")
        
        return episode_reward, turn_count

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
        agent_option = info.get("option", None)  # Get agent's chosen option
        target_exhibit = info.get("target_exhibit", None)  # For OfferTransition
        current_exhibit_completion = info.get("current_exhibit_completion", 0.0)  # For transition probability
        
        if agent_utterance:
            # Generate user response (now with transition logic)
            user_response = self.simulator.generate_user_response(
                agent_utterance, 
                agent_option=agent_option,
                target_exhibit=target_exhibit,
                current_exhibit_completion=current_exhibit_completion
            )
            
            # Track simulator LLM timing
            sim_time = user_response.get('simulator_llm_time', 0.0)
            if sim_time > 0:
                self.simulator_llm_times.append(sim_time)

            # Update environment with utterance and response type
            if user_response.get("utterance") is not None:
                self.env.update_user_state(
                    utterance=user_response["utterance"],
                    response_type=user_response.get("response_type", "statement")
                )

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
            
            # Record successful transition if one occurred (for 3-turn exemption rule)
            if info.get("option") == "OfferTransition" and user_response.get("transition_success", False):
                self.env.record_successful_transition()

    def _get_simulator_data(self) -> Dict[str, Any]:
        """Get current simulator data for logging"""
        return {
            "aoi": self.simulator.get_current_aoi(),
            "persona": getattr(self.simulator, 'current_persona', 'Unknown'),
            "utterance": "",
            "dwell_time": 0.5,
            "engagement_level": 0.5
        }

    def _print_state_vector(self, state: np.ndarray):
        """Print state vector at the beginning of turn (before action selection)"""
        if state is None:
            return
        
        # Calculate dimensions dynamically based on environment
        n_exhibits = self.env.n_exhibits
        focus_dim = n_exhibits + 1
        history_dim = n_exhibits + len(self.env.options)
        intent_dim = 64
        context_dim = 64
        total_dim = focus_dim + history_dim + intent_dim + context_dim
        
        print(f"📊 STATE VECTOR ({total_dim}-d) [Before Action Selection]:")
        
        # Focus vector
        focus_end = focus_dim
        focus_vec = state[0:focus_end]
        focus_idx = np.argmax(focus_vec)
        if focus_idx == n_exhibits:  # Last index is "no focus"
            focus_name = "No focus"
        else:
            focus_name = self.env.exhibit_keys[focus_idx]
        print(f"   👁️  Focus (0-{focus_end-1}): {focus_vec.round(2)} → {focus_name}")
        
        # History vector - Exhibit completion ratios
        completion_start = focus_end
        completion_end = completion_start + n_exhibits
        completion_vec = state[completion_start:completion_end]
        print(f"   📈 Completion ({completion_start}-{completion_end-1}):")
        for i, exhibit_name in enumerate(self.env.exhibit_keys):
            print(f"      [{i}] {exhibit_name:15s}: {completion_vec[i]:.2f}")
        
        # History vector - Option usage
        option_start = completion_end
        option_end = option_start + len(self.env.options)
        option_vec = state[option_start:option_end]
        print(f"   🎯 Option Usage ({option_start}-{option_end-1}):")
        for i, opt_name in enumerate(self.env.options):
            print(f"      [{i}] {opt_name:15s}: {option_vec[i]:.3f}")
        
        # Intent embedding
        intent_start = option_end
        intent_end = intent_start + intent_dim
        intent_vec = state[intent_start:intent_end]
        print(f"   🧠 Intent Emb ({intent_start}-{intent_end-1}): [{intent_vec[0]:.3f}, {intent_vec[1]:.3f}, ..., {intent_vec[-1]:.3f}] ({intent_dim}-d)")
        
        # Context embedding
        context_start = intent_end
        context_end = context_start + context_dim
        context_vec = state[context_start:context_end]
        print(f"   💬 Context Emb ({context_start}-{context_end-1}): [{context_vec[0]:.3f}, {context_vec[1]:.3f}, ..., {context_vec[-1]:.3f}] ({context_dim}-d)")
        print()
    
    def _print_agent_decision(self, action: Dict[str, Any], info: Dict[str, Any]):
        """Print agent decision and utterance (after action execution)"""
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
        
        # Show simulator engagement metrics
        engagement_level = user_response.get("engagement_level", 1.0)
        off_topic_strikes = user_response.get("off_topic_strikes", 0)
        agent_option = user_response.get("agent_option", "Unknown")
        response_type = user_response.get("response_type", "unknown")
        transition_success = user_response.get("transition_success", None)
        
        transition_status = ""
        if agent_option == "OfferTransition" and transition_success is not None:
            transition_status = f" | Transition: {'✅ SUCCESS' if transition_success else '❌ REJECTED'}"
        
        print(f"   🎯 Agent Option: {agent_option} | Response Type: {response_type}{transition_status} | Engagement: {engagement_level:.2f} | Strikes: {off_topic_strikes}")
        print()
        
        # REWARD BREAKDOWN SECTION
        print("💰 REWARD BREAKDOWN:")
        engagement_reward = info.get("reward_engagement", 0.0)
        novelty_reward = info.get("reward_novelty", 0.0)
        responsiveness_reward = info.get("reward_responsiveness", 0.0)
        conclude_bonus = info.get("reward_conclude", 0.0)
        transition_penalty = info.get("reward_transition_penalty", 0.0)
        spam_penalty = info.get("reward_spam_penalty", 0.0)
        
        print(f"   🔥 Engagement:    {engagement_reward:+7.3f} (lagged from prev turn)")
        print(f"   ✨ Novelty:       {novelty_reward:+7.3f} (verified facts)")
        if responsiveness_reward != 0:
            if responsiveness_reward > 0:
                print(f"   ✅ Responsiveness: {responsiveness_reward:+7.3f} (answered question)")
            else:
                print(f"   ❌ Responsiveness: {responsiveness_reward:+7.3f} (deflected question)")
        if conclude_bonus > 0:
            print(f"   🎯 Conclude:      {conclude_bonus:+7.3f} (exhibits covered)")
        if transition_penalty < 0:
            print(f"   ⚠️  Transition:    {transition_penalty:+7.3f} (insufficient facts)")
        if spam_penalty < 0:
            print(f"   🚫 Spam:          {spam_penalty:+7.3f} (too many transitions)")
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
        # DISABLED: Don't terminate early for thesis-level experiments
        # We need complete training runs for proper analysis
        return False

    def _save_checkpoint(self, checkpoint_name: str):
        """Save incremental checkpoint (metrics + model) for crash recovery - ALWAYS SAVE."""
        try:
            experiment_dir = os.environ.get('EXPERIMENT_DIR', None)
            if experiment_dir:
                checkpoint_dir = os.path.join(experiment_dir, 'checkpoints')
            else:
                checkpoint_dir = "training_logs/checkpoints"
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # Save metrics (ALWAYS)
            metrics_file = os.path.join(checkpoint_dir, f"{checkpoint_name}_metrics.json")
            self.metrics_tracker.save_to_json(metrics_file)
            
            # Save model if using Actor-Critic (ALWAYS)
            if self.use_actor_critic and hasattr(self, 'agent'):
                model_file = os.path.join(checkpoint_dir, f"{checkpoint_name}_model.pt")
                self.trainer.save_checkpoint(model_file, self.current_episode)
            
            print(f"   💾 Checkpoint saved: {checkpoint_name}")
            
        except Exception as e:
            print(f"   ⚠️  Failed to save checkpoint: {e}")
    
    def _print_progress_report(self):
        """Print comprehensive progress report every 50 episodes"""
        elapsed_time = time.time() - self.start_time if hasattr(self, 'start_time') else 0
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        
        # Calculate recent performance
        recent_50 = self.episode_rewards[-50:] if len(self.episode_rewards) >= 50 else self.episode_rewards
        avg_recent = np.mean(recent_50) if recent_50 else 0
        
        # Calculate overall trend
        if len(self.episode_rewards) >= 100:
            first_50 = np.mean(self.episode_rewards[:50])
            trend = "📈 Improving" if avg_recent > first_50 else "📊 Stable" if abs(avg_recent - first_50) < 0.5 else "📉 Declining"
        else:
            trend = "📊 Building baseline"
        
        # Estimate completion time
        if elapsed_time > 0 and len(self.episode_rewards) > 0:
            # Calculate average turns per episode so far
            total_turns_so_far = sum([len([h for h in self.monitor.training_history if h.get('episode') == ep]) 
                                     for ep in range(1, self.current_episode + 1)])
            avg_turns_per_episode = total_turns_so_far / self.current_episode if self.current_episode > 0 else self.max_turns_per_episode
            
            # Estimate remaining time based on turns (realistic timing)
            remaining_episodes = self.max_episodes - self.current_episode
            remaining_turns = remaining_episodes * avg_turns_per_episode
            eta_seconds = remaining_turns * 1.2  # More realistic: 1.2 seconds per turn
            eta_hours = eta_seconds / 3600
            eta_h = int(eta_hours)
            eta_m = int((eta_hours % 1) * 60)
            
            # Calculate current pace
            turns_per_hour = total_turns_so_far / (elapsed_time / 3600) if elapsed_time > 0 else 0
        else:
            eta_h = eta_m = 0
            turns_per_hour = 0
        
        print("\n" + "🔥" * 60)
        print(f"🌙 OVERNIGHT PROGRESS CHECKPOINT - Episode {self.current_episode:,}/{self.max_episodes:,}")
        print("🔥" * 60)
        print(f"⏰ Elapsed: {hours}h {minutes}m | ETA: {eta_h}h {eta_m}m")
        print(f"📊 Recent 50 Episodes Avg Reward: {avg_recent:.3f}")
        print(f"📈 Learning Trend: {trend}")
        print(f"� Current Pace: {turns_per_hour:.0f} turns/hour")
        print(f"�🗺️  Maps Saved: {(self.current_episode // self.map_interval)} animations")
        print(f"💾 Progress: {(self.current_episode/self.max_episodes)*100:.1f}% complete")
        print("🔥" * 60 + "\n")

    def _finalize_training(self):
        """Finalize training and save results"""
        print("\n" + "=" * 80)
        print("🏁 OVERNIGHT TRAINING SESSION COMPLETE!")
        print("=" * 80)
        
        # Calculate total training time
        training_time = time.time() - self.start_time if hasattr(self, 'start_time') else 0
        hours = int(training_time // 3600)
        minutes = int((training_time % 3600) // 60)
        
        # Update final statistics
        self.total_episodes = self.current_episode
        self.total_turns = len(self.monitor.training_history)
        
        # Get training summary
        summary = self.monitor.get_training_summary()
        
        # Print comprehensive statistics
        print(f"🎯 TRAINING PERFORMANCE:")
        print(f"   ├─ Total Episodes: {summary['total_episodes']:,}")
        print(f"   ├─ Total Turns: {summary['total_turns']:,}")
        print(f"   ├─ Training Time: {hours}h {minutes}m")
        print(f"   ├─ Episodes/Hour: {summary['total_episodes']/max(training_time/3600, 0.1):.1f}")
        print(f"   └─ Turns/Hour: {summary['total_turns']/max(training_time/3600, 0.1):.1f}")
        
        print(f"\n📊 LEARNING METRICS:")
        print(f"   ├─ Avg Reward/Turn: {summary['average_reward_per_turn']:.4f}")
        print(f"   ├─ Avg Reward/Episode: {summary['average_reward_per_episode']:.4f}")
        print(f"   ├─ Best Episode Reward: {summary['best_episode_reward']:.4f}")
        print(f"   ├─ Final 100 Episodes Avg: {np.mean(self.episode_rewards[-100:]):.4f}")
        print(f"   └─ Learning Curve Trend: {'📈 Improving' if len(self.episode_rewards) > 100 and np.mean(self.episode_rewards[-100:]) > np.mean(self.episode_rewards[:100]) else '📉 Stable/Declining'}")
        
        print(f"\n🎮 STRATEGY ANALYSIS:")
        print(f"   ├─ Most Used Option: {summary['most_used_option']}")
        print(f"   ├─ Most Used Subaction: {summary['most_used_subaction']}")
        print(f"   └─ Strategy Diversity: {len(set([h['option'] for h in self.monitor.training_history if 'option' in h]))} unique options used")
        
        # Map generation summary
        maps_saved = (self.current_episode // self.map_interval) if self.enable_map_viz else 0
        print(f"\n🗺️  VISUALIZATION:")
        print(f"   ├─ Map Episodes Saved: {maps_saved} (every {self.map_interval} episodes)")
        print(f"   ├─ Total Frames Captured: {len(self.map_visualizer.frames) if self.enable_map_viz else 0}")
        print(f"   └─ Storage Optimization: {'✓ Enabled' if maps_saved < self.current_episode else '✗ Disabled'}")
        
        # Generate comprehensive RL analysis report (ENABLED for thesis)
        # Only generate if we have enough data
        if len(self.episode_rewards) >= 50:
            self._generate_rl_analysis_report(training_time, summary)
        
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
        
        # Save training log (ALWAYS SAVE)
        self._save_training_log()
        
        # ===== SAVE METRICS (ALWAYS SAVE FOR THESIS) =====
        print("\n💾 Saving comprehensive metrics...")
        
        # Determine save directory
        experiment_dir = os.environ.get('EXPERIMENT_DIR', None)
        if experiment_dir:
            logs_dir = os.path.join(experiment_dir, 'logs')
            os.makedirs(logs_dir, exist_ok=True)
        else:
            logs_dir = "training_logs"
            os.makedirs(logs_dir, exist_ok=True)
        
        # Save from live monitor (only if enabled and has data)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if self.enable_live_monitor and (self.live_monitor.episode_metrics or self.live_monitor.turn_metrics):
            if experiment_dir:
                from pathlib import Path
                self.live_monitor.log_dir = Path(logs_dir)
            self.live_monitor.save_metrics(f"monitor_{timestamp}")
        
        # Save from metrics tracker (ALWAYS SAVE)
        if experiment_dir:
            metrics_file = os.path.join(logs_dir, f"metrics_tracker_{timestamp}.json")
        else:
            metrics_file = os.path.join(logs_dir, f"metrics_tracker_{timestamp}.json")
        self.metrics_tracker.save_to_json(metrics_file)
            
        # Print summary statistics (ALWAYS)
        summary = self.metrics_tracker.get_summary_statistics()
        print("\n📊 Training Metrics Summary:")
        print(f"   Mean Return: {summary.get('mean_return', 0.0):.3f} ± {summary.get('std_return', 0.0):.3f}")
        print(f"   Recent Mean Return (last 100): {summary.get('recent_mean_return', 0.0):.3f}")
        print(f"   Mean Episode Length: {summary.get('mean_length', 0.0):.1f} turns")
        print(f"   Mean Coverage: {summary.get('mean_coverage', 0.0):.1%}")
        print(f"   Transition Success Rate: {summary.get('transition_success_rate', 0.0):.1%}")
        print(f"   Question Answer Rate: {summary.get('question_answer_rate', 0.0):.1%}")
        
        option_usage = summary.get('option_usage', {})
        if option_usage:
            print(f"\n   Option Usage:")
            for opt, prop in sorted(option_usage.items(), key=lambda x: -x[1]):
                print(f"      {opt}: {prop:.1%}")
        
        # ===== SAVE FINAL MODEL =====
        if self.use_actor_critic and hasattr(self, 'agent'):
            print("\n🎓 Saving final trained model...")
            experiment_dir = os.environ.get('EXPERIMENT_DIR', None)
            if experiment_dir:
                model_dir = os.path.join(experiment_dir, 'models')
                os.makedirs(model_dir, exist_ok=True)
                final_model_path = os.path.join(model_dir, f'final_model_ep{self.current_episode}.pt')
                self.trainer.save_checkpoint(final_model_path, self.current_episode)
                print(f"   ✅ Final model saved: {final_model_path}")
            else:
                # Fallback to default location
                os.makedirs('models', exist_ok=True)
                final_model_path = f'models/final_model_ep{self.current_episode}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pt'
                self.trainer.save_checkpoint(final_model_path, self.current_episode)
                print(f"   ✅ Final model saved: {final_model_path}")
        
        # Clean up
        self.monitor.close()
        self.map_visualizer.close()
        print("\n✅ Training completed successfully!")

    def _generate_rl_analysis_report(self, training_time, summary):
        """Generate comprehensive RL analysis report with convergence analysis"""
        print("\n" + "🧠" * 80)
        print(" " * 25 + "REINFORCEMENT LEARNING ANALYSIS REPORT" + " " * 25)
        print("🧠" * 80)
        
        # Sample Efficiency Analysis
        print(f"\n📈 SAMPLE EFFICIENCY:")
        total_samples = len(self.monitor.training_history)
        episodes_to_convergence = self._analyze_convergence()
        print(f"   ├─ Total Samples (Turns): {total_samples:,}")
        print(f"   ├─ Sample Efficiency: {total_samples/self.current_episode:.1f} turns/episode")
        print(f"   ├─ Convergence Episode: {episodes_to_convergence if episodes_to_convergence else 'Not achieved'}")
        print(f"   └─ Samples to Convergence: {episodes_to_convergence * (total_samples/self.current_episode) if episodes_to_convergence else 'N/A'}")
        
        # Learning Curve Analysis
        print(f"\n📊 LEARNING CURVE ANALYSIS:")
        if len(self.episode_rewards) >= 100:
            early_performance = np.mean(self.episode_rewards[:100])
            late_performance = np.mean(self.episode_rewards[-100:])
            improvement = late_performance - early_performance
            improvement_rate = improvement / len(self.episode_rewards) * 100
            
            print(f"   ├─ Early Performance (first 100): {early_performance:.3f}")
            print(f"   ├─ Late Performance (last 100): {late_performance:.3f}")
            print(f"   ├─ Total Improvement: {improvement:.3f}")
            print(f"   ├─ Improvement Rate: {improvement_rate:.4f}/100 episodes")
            print(f"   └─ Learning Stability: {'✓ Stable' if np.std(self.episode_rewards[-100:]) < 2.0 else '⚠ Unstable'}")
        
        # Policy Analysis
        print(f"\n🎯 POLICY ANALYSIS:")
        option_counts = {}
        termination_counts = {}
        option_durations = []
        
        current_option_start = 0
        for i, turn in enumerate(self.monitor.training_history):
            option = turn.get('option', 'Unknown')
            option_counts[option] = option_counts.get(option, 0) + 1
            
            # Track option terminations and durations
            if turn.get('option_terminated', False):
                duration = i - current_option_start + 1
                option_durations.append(duration)
                termination_counts[option] = termination_counts.get(option, 0) + 1
                current_option_start = i + 1
        
        print(f"   ├─ Option Usage Distribution:")
        for option, count in sorted(option_counts.items(), key=lambda x: x[1], reverse=True):
            pct = count / len(self.monitor.training_history) * 100
            print(f"   │    • {option}: {count:,} turns ({pct:.1f}%)")
        
        if option_durations:
            print(f"   ├─ Average Option Duration: {np.mean(option_durations):.1f} turns")
            print(f"   ├─ Option Duration Std: {np.std(option_durations):.1f}")
            print(f"   └─ Termination Learning: {'✓ Active' if len(termination_counts) > 0 else '⚠ No terminations detected'}")
        
        # Exploration vs Exploitation Analysis
        print(f"\n🎮 EXPLORATION-EXPLOITATION BALANCE:")
        if len(self.episode_rewards) >= 200:
            # Convert deque to list for slicing
            history_list = list(self.monitor.training_history)
            quarter_size = len(history_list) // 4
            entropy_early = self._calculate_action_entropy(history_list[:quarter_size])
            entropy_late = self._calculate_action_entropy(history_list[-quarter_size:])
            print(f"   ├─ Early Action Entropy: {entropy_early:.3f}")
            print(f"   ├─ Late Action Entropy: {entropy_late:.3f}")
            print(f"   ├─ Exploration Decay: {entropy_early - entropy_late:.3f}")
            print(f"   └─ Exploitation Status: {'✓ Good balance' if 0.1 < entropy_late < entropy_early else '⚠ Check entropy schedule'}")
        
        # Credit Assignment Analysis (HRL specific)
        print(f"\n🔗 HIERARCHICAL CREDIT ASSIGNMENT:")
        option_rewards = {}
        for turn in self.monitor.training_history:
            option = turn['info'].get('option', 'Unknown')  # Fixed: get option from 'info' dict
            reward = turn.get('reward', 0)
            if option not in option_rewards:
                option_rewards[option] = []
            option_rewards[option].append(reward)
        
        for option, rewards in option_rewards.items():
            if rewards:
                avg_reward = np.mean(rewards)
                print(f"   ├─ {option} Average Reward: {avg_reward:.3f}")
        
        # Generate and save convergence plots
        self._generate_convergence_plots()
        
        # Generate comprehensive analysis with individual plots
        self._generate_comprehensive_analysis()
        
        print("🧠" * 80)

    def _analyze_convergence(self):
        """Analyze when the policy converged"""
        if len(self.episode_rewards) < 200:
            return None
        
        # Use sliding window approach to detect convergence
        window_size = 50
        threshold = 0.05  # 5% change threshold
        
        for i in range(100, len(self.episode_rewards) - window_size):
            current_window = self.episode_rewards[i:i+window_size]
            next_window = self.episode_rewards[i+window_size//2:i+window_size//2+window_size]
            
            current_mean = np.mean(current_window)
            next_mean = np.mean(next_window)
            
            if abs(next_mean - current_mean) / (abs(current_mean) + 1e-8) < threshold:
                # Check if this stability continues
                stable_count = 0
                for j in range(i, min(i+100, len(self.episode_rewards)-window_size)):
                    test_window = self.episode_rewards[j:j+window_size]
                    if abs(np.mean(test_window) - current_mean) / (abs(current_mean) + 1e-8) < threshold:
                        stable_count += 1
                
                if stable_count > 50:  # Stable for 50+ episodes
                    return i
        
        return None

    def _calculate_action_entropy(self, history):
        """Calculate action entropy for exploration analysis"""
        actions = [turn.get('subaction', 'Unknown') for turn in history]
        action_counts = {}
        for action in actions:
            action_counts[action] = action_counts.get(action, 0) + 1
        
        total = len(actions)
        entropy = 0
        for count in action_counts.values():
            p = count / total
            if p > 0:
                entropy -= p * np.log2(p)
        
        return entropy

    def _generate_convergence_plots(self):
        """Generate comprehensive convergence and learning analysis plots"""
        try:
            import matplotlib.pyplot as plt
            
            # Create comprehensive analysis figure
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            # Plot 1: Learning Curve with Moving Averages
            episodes = list(range(1, len(self.episode_rewards) + 1))
            ax1.plot(episodes, self.episode_rewards, alpha=0.3, color='lightblue', label='Episode Rewards')
            
            # Moving averages
            if len(self.episode_rewards) >= 50:
                ma_50 = []
                for i in range(len(self.episode_rewards)):
                    start = max(0, i - 49)
                    ma_50.append(np.mean(self.episode_rewards[start:i+1]))
                ax1.plot(episodes, ma_50, color='orange', linewidth=2, label='50-Episode MA')
            
            if len(self.episode_rewards) >= 100:
                ma_100 = []
                for i in range(len(self.episode_rewards)):
                    start = max(0, i - 99)
                    ma_100.append(np.mean(self.episode_rewards[start:i+1]))
                ax1.plot(episodes, ma_100, color='red', linewidth=2, label='100-Episode MA')
            
            convergence_episode = self._analyze_convergence()
            if convergence_episode:
                ax1.axvline(x=convergence_episode, color='green', linestyle='--', linewidth=2, label=f'Convergence (ep {convergence_episode})')
            
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Reward')
            ax1.set_title('Learning Curve Analysis')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Convergence Analysis
            if len(self.episode_rewards) >= 100:
                convergence_analysis = []
                window = 100
                for i in range(window, len(self.episode_rewards)):
                    recent_performance = np.mean(self.episode_rewards[i-window:i])
                    convergence_analysis.append(recent_performance)
                
                conv_episodes = list(range(window, len(self.episode_rewards)))
                ax2.plot(conv_episodes, convergence_analysis, 'go-', linewidth=2)
                ax2.set_xlabel('Episode')
                ax2.set_ylabel('100-Episode Moving Average')
                ax2.set_title('Convergence Analysis')
                ax2.grid(True, alpha=0.3)
                
                if convergence_episode:
                    ax2.axvline(x=convergence_episode, color='red', linestyle='--', label='Detected Convergence')
                    ax2.legend()
            
            # Plot 3: Option Usage Distribution
            option_counts = {}
            for turn in self.monitor.training_history:
                option = turn.get('option', 'Unknown')
                option_counts[option] = option_counts.get(option, 0) + 1
            
            if option_counts:
                options = list(option_counts.keys())
                counts = list(option_counts.values())
                colors = plt.cm.Set3(np.linspace(0, 1, len(options)))
                
                wedges, texts, autotexts = ax3.pie(counts, labels=options, autopct='%1.1f%%', colors=colors)
                ax3.set_title('Option Usage Distribution')
            
            # Plot 4: Training Stability Analysis
            if len(self.episode_rewards) >= 200:
                stability_window = 50
                stability_scores = []
                
                for i in range(stability_window, len(self.episode_rewards)):
                    window_data = self.episode_rewards[i-stability_window:i]
                    stability = 1.0 / (1.0 + np.std(window_data))  # Higher = more stable
                    stability_scores.append(stability)
                
                stability_episodes = list(range(stability_window, len(self.episode_rewards)))
                ax4.plot(stability_episodes, stability_scores, 'b-', linewidth=2)
                ax4.set_xlabel('Episode')
                ax4.set_ylabel('Stability Score')
                ax4.set_title('Training Stability Analysis')
                ax4.grid(True, alpha=0.3)
                ax4.set_ylim(0, 1)
            
            plt.tight_layout()
            
            # Save plots
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_filename = f"training_logs/rl_analysis_{timestamp}.png"
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            
            print(f"\n📊 RL Analysis plots saved to: {plot_filename}")
            
            plt.close()
            
        except ImportError:
            print("⚠️  Matplotlib not available - skipping plot generation")
        except Exception as e:
            print(f"⚠️  Failed to generate plots: {e}")

    def _generate_comprehensive_analysis(self):
        """Generate comprehensive analysis with individual and combined plots"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec
            
            experiment_dir = os.environ.get('EXPERIMENT_DIR', None)
            if not experiment_dir:
                print("⚠️  No experiment directory found - skipping comprehensive analysis")
                return
            
            plots_dir = os.path.join(experiment_dir, 'plots')
            os.makedirs(plots_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            print("\n📊 Generating Comprehensive Analysis Plots...")
            
            # ===== INDIVIDUAL PLOTS =====
            
            # 1. Learning Curve (Individual)
            fig1, ax1 = plt.subplots(figsize=(12, 6))
            episodes = range(1, len(self.episode_rewards) + 1)
            ax1.plot(episodes, self.episode_rewards, alpha=0.3, color='blue', label='Raw Returns')
            
            # Moving average
            if len(self.episode_rewards) >= 50:
                window = 50
                moving_avg = np.convolve(self.episode_rewards, np.ones(window)/window, mode='valid')
                ax1.plot(range(window, len(self.episode_rewards) + 1), moving_avg, 
                        linewidth=3, color='red', label='50-Episode Moving Average')
            
            ax1.set_xlabel('Episode', fontsize=14)
            ax1.set_ylabel('Total Reward', fontsize=14)
            ax1.set_title('Learning Curve', fontsize=16, fontweight='bold')
            ax1.legend(fontsize=12)
            ax1.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f'learning_curve_{timestamp}.png'), dpi=300, bbox_inches='tight')
            plt.close()
            print(f"   ✓ Saved: learning_curve_{timestamp}.png")
            
            # 2. Option Usage Distribution (Individual)
            fig2, ax2 = plt.subplots(figsize=(10, 8))
            option_counts = {}
            for turn in self.monitor.training_history:
                option = turn['info'].get('option', 'Unknown')
                option_counts[option] = option_counts.get(option, 0) + 1
            
            if option_counts:
                options = list(option_counts.keys())
                counts = list(option_counts.values())
                colors = plt.cm.Set3(np.linspace(0, 1, len(options)))
                
                wedges, texts, autotexts = ax2.pie(counts, labels=options, autopct='%1.1f%%',
                                                   colors=colors, textprops={'fontsize': 12})
                ax2.set_title('Option Usage Distribution (All Episodes)', fontsize=16, fontweight='bold')
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, f'option_distribution_{timestamp}.png'), dpi=300, bbox_inches='tight')
                plt.close()
                print(f"   ✓ Saved: option_distribution_{timestamp}.png")
            
            # 3. Option Evolution Over Time (NEW!)
            fig3, ax3 = plt.subplots(figsize=(14, 8))
            
            # Calculate option usage in windows
            window_size = max(10, len(self.episode_rewards) // 10)  # 10 windows
            option_evolution = {}
            
            # Convert deque to list for slicing
            history_list = list(self.monitor.training_history)
            for i in range(0, len(history_list), window_size):
                window = history_list[i:i+window_size]
                window_counts = {}
                for turn in window:
                    option = turn['info'].get('option', 'Unknown')
                    window_counts[option] = window_counts.get(option, 0) + 1
                
                # Normalize to percentages
                total = sum(window_counts.values())
                for option, count in window_counts.items():
                    if option not in option_evolution:
                        option_evolution[option] = []
                    option_evolution[option].append((count / total * 100) if total > 0 else 0)
            
            # Plot evolution for each option
            if option_evolution:
                window_episodes = [i * window_size / len(history_list) * len(self.episode_rewards)
                                 for i in range(len(next(iter(option_evolution.values()))))]
            else:
                window_episodes = []
            
            colors_map = {'Explain': 'blue', 'AskQuestion': 'green', 'OfferTransition': 'orange', 'Conclude': 'red'}
            for option, percentages in option_evolution.items():
                color = colors_map.get(option, 'gray')
                ax3.plot(window_episodes, percentages, marker='o', linewidth=2, 
                        label=option, color=color)
            
            ax3.set_xlabel('Episode', fontsize=14)
            ax3.set_ylabel('Usage Percentage (%)', fontsize=14)
            ax3.set_title('Option Usage Evolution Over Training', fontsize=16, fontweight='bold')
            ax3.legend(fontsize=12)
            ax3.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f'option_evolution_{timestamp}.png'), dpi=300, bbox_inches='tight')
            plt.close()
            print(f"   ✓ Saved: option_evolution_{timestamp}.png")
            
            # 4. Reward Distribution (Individual)
            fig4, ax4 = plt.subplots(figsize=(10, 6))
            ax4.hist(self.episode_rewards, bins=30, alpha=0.7, color='purple', edgecolor='black')
            ax4.axvline(np.mean(self.episode_rewards), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(self.episode_rewards):.2f}')
            ax4.set_xlabel('Total Reward', fontsize=14)
            ax4.set_ylabel('Frequency', fontsize=14)
            ax4.set_title('Reward Distribution', fontsize=16, fontweight='bold')
            ax4.legend(fontsize=12)
            ax4.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f'reward_distribution_{timestamp}.png'), dpi=300, bbox_inches='tight')
            plt.close()
            print(f"   ✓ Saved: reward_distribution_{timestamp}.png")
            
            # ===== COMBINED MEGA PLOT =====
            fig = plt.figure(figsize=(20, 12))
            gs = gridspec.GridSpec(3, 3, hspace=0.3, wspace=0.3)
            
            # Learning Curve (large, top left)
            ax1 = fig.add_subplot(gs[0, :2])
            ax1.plot(episodes, self.episode_rewards, alpha=0.3, color='blue')
            if len(self.episode_rewards) >= 50:
                ax1.plot(range(50, len(self.episode_rewards) + 1), moving_avg, linewidth=3, color='red')
            ax1.set_xlabel('Episode', fontsize=12)
            ax1.set_ylabel('Reward', fontsize=12)
            ax1.set_title('Learning Curve', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            
            # Option Distribution (top right)
            ax2 = fig.add_subplot(gs[0, 2])
            if option_counts:
                wedges, texts, autotexts = ax2.pie(counts, labels=options, autopct='%1.1f%%', colors=colors)
                ax2.set_title('Option Distribution', fontsize=12, fontweight='bold')
            
            # Option Evolution (middle, full width)
            ax3 = fig.add_subplot(gs[1, :])
            for option, percentages in option_evolution.items():
                color = colors_map.get(option, 'gray')
                ax3.plot(window_episodes, percentages, marker='o', linewidth=2, label=option, color=color)
            ax3.set_xlabel('Episode', fontsize=12)
            ax3.set_ylabel('Usage %', fontsize=12)
            ax3.set_title('Option Evolution Over Time', fontsize=14, fontweight='bold')
            ax3.legend(fontsize=10)
            ax3.grid(True, alpha=0.3)
            
            # Reward Distribution (bottom left)
            ax4 = fig.add_subplot(gs[2, 0])
            ax4.hist(self.episode_rewards, bins=20, alpha=0.7, color='purple', edgecolor='black')
            ax4.set_xlabel('Reward', fontsize=10)
            ax4.set_ylabel('Frequency', fontsize=10)
            ax4.set_title('Reward Distribution', fontsize=12, fontweight='bold')
            ax4.grid(True, alpha=0.3, axis='y')
            
            # Convergence Analysis (bottom middle)
            ax5 = fig.add_subplot(gs[2, 1])
            if len(self.episode_rewards) >= 10:
                rolling_best = np.maximum.accumulate(self.episode_rewards)
                ax5.plot(episodes, rolling_best, linewidth=2, color='green')
                ax5.set_xlabel('Episode', fontsize=10)
                ax5.set_ylabel('Best So Far', fontsize=10)
                ax5.set_title('Convergence', fontsize=12, fontweight='bold')
                ax5.grid(True, alpha=0.3)
            
            # Statistics (bottom right)
            ax6 = fig.add_subplot(gs[2, 2])
            ax6.axis('off')
            stats_text = f"Statistics\n{'='*30}\n\n"
            stats_text += f"Episodes: {len(self.episode_rewards)}\n"
            if len(self.episode_rewards) > 0:
                stats_text += f"Avg Reward: {np.mean(self.episode_rewards):.2f}\n"
                stats_text += f"Std Reward: {np.std(self.episode_rewards):.2f}\n"
                stats_text += f"Best Reward: {max(self.episode_rewards):.2f}\n"
                stats_text += f"Worst Reward: {min(self.episode_rewards):.2f}\n"
            else:
                stats_text += "No episodes completed\n"
            ax6.text(0.1, 0.9, stats_text, fontsize=11, verticalalignment='top',
                    family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plt.suptitle('Comprehensive Training Analysis', fontsize=18, fontweight='bold')
            plt.savefig(os.path.join(plots_dir, f'comprehensive_analysis_{timestamp}.png'), dpi=300, bbox_inches='tight')
            plt.close()
            print(f"   ✓ Saved: comprehensive_analysis_{timestamp}.png")
            
            print(f"\n✅ All analysis plots saved to: {plots_dir}")
            
        except Exception as e:
            print(f"⚠️  Failed to generate comprehensive analysis: {e}")
            import traceback
            traceback.print_exc()
    
    def _save_training_log(self):
        """Save training results to JSON file - ALWAYS SAVE to experiment directory"""
        try:
            # Determine save location (prefer experiment directory)
            experiment_dir = os.environ.get('EXPERIMENT_DIR', None)
            now = datetime.now()
            timestamp = now.strftime("%H%M%S")
            
            if experiment_dir:
                # Save to experiment directory
                logs_dir = os.path.join(experiment_dir, 'logs')
                os.makedirs(logs_dir, exist_ok=True)
                filename = f"training_log_{timestamp}.json"
                filepath = os.path.join(logs_dir, filename)
            else:
                # Fallback to date-organized directory
                date_folder = now.strftime("%Y-%m-%d")
                logs_dir = "training_logs"
                if not os.path.exists(logs_dir):
                    os.makedirs(logs_dir)
                
                date_dir = os.path.join(logs_dir, date_folder)
                if not os.path.exists(date_dir):
                    os.makedirs(date_dir)
                    print(f"📁 Created new log directory: {date_dir}")
                
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
