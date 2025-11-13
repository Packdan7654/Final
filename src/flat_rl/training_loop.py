import os
from typing import Dict, Any

from src.training.training_loop import HRLTrainingLoop
from src.visualization.museum_map_visualizer import MuseumMapVisualizer

from .agent import FlatActorCriticAgent
from .env import FlatDialogueEnv
from .trainer import FlatActorCriticTrainer


class FlatTrainingLoop(HRLTrainingLoop):
    """
    Training loop wrapper that reuses the hierarchical infrastructure but swaps
    in a flat policy/agent whose action space is the set of primitive subactions.
    """

    def __init__(self, *args, **kwargs):
        # Force hierarchical components to stay uninitialized in parent
        kwargs = dict(kwargs)
        kwargs.setdefault("use_actor_critic", False)
        self._flat_knowledge_graph_path = kwargs.get("knowledge_graph_path", None)
        super().__init__(*args, **kwargs)

        # Replace environment with flat variant (same simulator + rewards)
        max_turns = kwargs.get("max_turns_per_episode", self.max_turns_per_episode)
        self.env = FlatDialogueEnv(
            knowledge_graph_path=self._flat_knowledge_graph_path,
            max_turns=max_turns,
        )

        # Rebuild map visualizer with new environment reference
        map_save_dir = "training_logs/maps"
        experiment_dir = os.environ.get("EXPERIMENT_DIR", None)
        if experiment_dir:
            map_save_dir = os.path.join(experiment_dir, "maps")

        self.map_visualizer = MuseumMapVisualizer(
            enabled=self.enable_map_viz,
            exhibits=self.env.exhibit_keys,
            save_dir=map_save_dir,
            live_display=False,
        )

        # Initialize flat agent/trainer
        state_dim = self.env.observation_space.shape[0]
        self.agent = FlatActorCriticAgent(
            state_dim=state_dim,
            options=self.env.options,
            subactions=self.env.subactions,
            hidden_dim=256,
            lstm_hidden_dim=128,
            use_lstm=True,
            device=self.device,
        )

        learning_rate = kwargs.get("learning_rate", 1e-4)
        gamma = kwargs.get("gamma", 0.99)

        self.trainer = FlatActorCriticTrainer(
            agent=self.agent,
            learning_rate=learning_rate,
            gamma=gamma,
            device=self.device,
        )

        # Episode buffer mirrors base loop for compatibility
        self.use_actor_critic = True
        self.episode_buffer = {
            "states": [],
            "options": [],
            "subactions": [],
            "rewards": [],
            "next_states": [],
            "dones": [],
        }

        self.training_title = "FLAT RL MUSEUM DIALOGUE TRAINING"

    # ------------------------------------------------------------------ #
    # Action selection override
    # ------------------------------------------------------------------ #
    def _generate_action_actor_critic(self, obs) -> Dict[str, Any]:
        available_options = self.env._get_available_options()
        if not available_options:
            return {"option": 0, "subaction": 0, "terminate_option": False}

        available_subactions_dict = {
            opt: self.env._get_available_subactions(opt) for opt in available_options
        }

        action_info = self.agent.select_action(
            state=obs,
            available_options=available_options,
            available_subactions_dict=available_subactions_dict,
            deterministic=False,
        )

        return {
            "option": action_info["option"],
            "option_name": action_info["option_name"],
            "subaction": action_info["subaction"],
            "subaction_name": action_info["subaction_name"],
            "terminate_option": False,
            "flat_action": action_info["flat_action"],
        }
    
    # ------------------------------------------------------------------ #
    # Override _run_episode to handle flat action space
    # ------------------------------------------------------------------ #
    def _run_episode(self):
        """Override to pass flat_action integer to env.step() instead of dict."""
        # Get initial observation
        obs, info = self.env.reset()
        
        # Reset agent state
        if self.use_actor_critic:
            self.agent.reset()
        
        # Reset episode buffer
        self.episode_buffer = {
            "states": [],
            "options": [],
            "subactions": [],
            "rewards": [],
            "next_states": [],
            "dones": [],
        }
        
        # Reset map visualizer
        self.map_visualizer.reset()
        
        episode_reward = 0.0
        turn_count = 0
        
        # Track reward components for episode summary
        episode_reward_components = {
            "engagement": 0.0,
            "novelty": 0.0,
            "responsiveness": 0.0,
            "transition": 0.0,
            "conclude": 0.0
        }
        
        # Episode loop
        while turn_count < self.max_turns_per_episode:
            turn_count += 1
            
            # Update environment with simulator state
            self._update_environment_state()
            
            # Generate agent action
            if self.use_actor_critic:
                action_dict = self._generate_action_actor_critic(obs)
                # Extract flat_action integer for flat environment
                flat_action = action_dict.get("flat_action", 0)
            else:
                # Random action for flat space
                flat_action = self.env.action_space.sample()
                action_dict = {"flat_action": flat_action}
            
            # Execute environment step with flat action integer
            next_obs, reward, done, truncated, info = self.env.step(flat_action)
            
            # Store transition (for compatibility, store as if hierarchical)
            self.episode_buffer["states"].append(obs)
            self.episode_buffer["options"].append(action_dict.get("option", 0))
            self.episode_buffer["subactions"].append(action_dict.get("subaction", 0))
            self.episode_buffer["rewards"].append(reward)
            self.episode_buffer["next_states"].append(next_obs)
            self.episode_buffer["dones"].append(done or truncated)
            
            # Get simulator data for metrics tracking
            simulator_state = self.simulator.get_current_state()
            user_response = simulator_state.get("last_user_response", {})
            
            # Build turn_data for metrics tracking (include flat_action_name)
            turn_data = {
                "agent_utterance": info.get("agent_utterance", ""),
                "user_utterance": user_response.get("utterance", ""),
                "option": info.get("option", "Unknown"),  # For compatibility
                "subaction": info.get("subaction", "Unknown"),  # For compatibility
                "flat_action_name": info.get("flat_action_name", ""),  # Flat RL specific
                "dwell": user_response.get("gaze_features", [0.0])[0],
                "response_type": user_response.get("response_type", "unknown"),
                "current_exhibit": info.get("current_exhibit", "Unknown"),
                "facts_shared": info.get("facts_shared", 0),
                "reward_engagement": info.get("reward_engagement", 0.0),
                "reward_novelty": info.get("reward_novelty", 0.0),
                "reward_responsiveness": info.get("reward_responsiveness", 0.0),
                "reward_transition_insufficiency": info.get("reward_transition_insufficiency", 0.0),
                "reward_transition_sufficiency": info.get("reward_transition_sufficiency", 0.0),
                "reward_transition_frequency": info.get("reward_transition_frequency", 0.0),
                "reward_conclude": info.get("reward_conclude", 0.0),
                "total_reward": reward,
                "exhibit_coverage": self.env._get_museum_exhibit_coverage()
            }
            
            # Update metrics tracker with turn data (includes flat_action_name)
            self.metrics_tracker.update_turn(turn_data)
            
            # Accumulate reward components for this episode
            episode_reward_components["engagement"] += turn_data.get("reward_engagement", 0.0)
            episode_reward_components["novelty"] += turn_data.get("reward_novelty", 0.0)
            episode_reward_components["responsiveness"] += turn_data.get("reward_responsiveness", 0.0)
            # Sum all transition-related rewards
            episode_reward_components["transition"] += (
                turn_data.get("reward_transition_insufficiency", 0.0) +
                turn_data.get("reward_transition_sufficiency", 0.0) +
                turn_data.get("reward_transition_frequency", 0.0)
            )
            episode_reward_components["conclude"] += turn_data.get("reward_conclude", 0.0)
            
            episode_reward += reward
            obs = next_obs
            
            if done or truncated:
                break
        
        # Update trainer if using actor-critic
        if self.use_actor_critic and len(self.episode_buffer["states"]) > 0:
            train_stats = self.trainer.update(
                states=self.episode_buffer["states"],
                options=self.episode_buffer["options"],
                subactions=self.episode_buffer["subactions"],
                rewards=self.episode_buffer["rewards"],
                next_states=self.episode_buffer["next_states"],
                dones=self.episode_buffer["dones"]
            )
            
            # Update metrics tracker with RL training stats
            if train_stats:
                self.metrics_tracker.update_training_stats(train_stats)
        
        # Build episode summary for metrics
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
            "mean_value": train_stats.get('mean_value', 0.0) if train_stats else 0.0,
            "reward_engagement": episode_reward_components["engagement"],
            "reward_novelty": episode_reward_components["novelty"],
            "reward_responsiveness": episode_reward_components["responsiveness"],
            "reward_transition": episode_reward_components["transition"],
            "reward_conclude": episode_reward_components["conclude"]
        }
        
        # Update metrics tracker with episode summary
        self.metrics_tracker.update_episode(episode_summary)
        
        return episode_reward, turn_count

__all__ = ["FlatTrainingLoop"]


