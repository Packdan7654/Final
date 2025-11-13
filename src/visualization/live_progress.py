"""
Clean live progress tracker for training
Shows episode-level stats without verbose turn-by-turn details
"""

import sys
from collections import defaultdict
from datetime import datetime, timedelta

class LiveProgressTracker:
    """Clean, real-time training progress display"""
    
    def __init__(self, max_episodes):
        self.max_episodes = max_episodes
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_times = []
        self.option_counts = defaultdict(int)
        self.current_episode = 0
        self.current_ep_reward = 0.0
        self.current_ep_turns = 0
        self.start_time = datetime.now()
        
    def start_episode(self, episode_num):
        """Start tracking a new episode"""
        self.current_episode = episode_num
        self.current_ep_reward = 0.0
        self.current_ep_turns = 0
    
    def update_turn(self, reward, option):
        """Update with turn-level info (no printing)"""
        self.current_ep_reward += reward
        self.current_ep_turns += 1
        self.option_counts[option] += 1
    
    def end_episode(self, total_reward, length, episode_time=None):
        """End episode and show summary"""
        self.episode_rewards.append(total_reward)
        self.episode_lengths.append(length)
        if episode_time is not None:
            # Store episode time if provided
            if not hasattr(self, 'episode_times'):
                self.episode_times = []
            self.episode_times.append(episode_time)
        self._print_episode_summary()
    
    def _print_episode_summary(self):
        """Print clean episode summary"""
        # Calculate stats
        recent_window = min(10, len(self.episode_rewards))
        recent_avg = sum(self.episode_rewards[-recent_window:]) / recent_window if recent_window > 0 else 0.0
        
        # Elapsed time
        elapsed = (datetime.now() - self.start_time).total_seconds()
        hours, remainder = divmod(int(elapsed), 3600)
        minutes, seconds = divmod(remainder, 60)
        
        # Episode progress
        progress_pct = (self.current_episode / self.max_episodes) * 100
        
        # Calculate ETA and finish time
        if elapsed > 0 and self.current_episode > 0:
            eps_per_sec = self.current_episode / elapsed
            remaining_eps = self.max_episodes - self.current_episode
            eta_sec = remaining_eps / eps_per_sec if eps_per_sec > 0 else 0
            eta_hours = int(eta_sec // 3600)
            eta_mins = int((eta_sec % 3600) // 60)
            eta_secs = int(eta_sec % 60)
            
            # Calculate finish time
            finish_time = datetime.now() + timedelta(seconds=eta_sec)
            finish_str = finish_time.strftime("%H:%M:%S")
            eta_str = f"{eta_hours:02d}:{eta_mins:02d}:{eta_secs:02d}"
        else:
            eta_str = "??:??:??"
            finish_str = "??:??:??"
        
        # Option distribution (top 3 cumulative)
        total_turns = sum(self.option_counts.values())
        option_dist = {k: (v/total_turns*100) for k, v in self.option_counts.items()} if total_turns > 0 else {}
        top_options = sorted(option_dist.items(), key=lambda x: -x[1])[:3]
        options_str = ', '.join([f'{k}({v:.0f}%)' for k, v in top_options])
        
        # Current episode turns (not cumulative)
        ep_turns = self.episode_lengths[-1]
        
        # Episode time (time for this specific episode)
        if len(self.episode_times) > 0:
            ep_time_sec = self.episode_times[-1]
            ep_time_mins = int(ep_time_sec // 60)
            ep_time_secs = int(ep_time_sec % 60)
            ep_time_str = f"{ep_time_mins:02d}:{ep_time_secs:02d}"
        else:
            ep_time_str = "??:??"
        
        # Print on new line (no overwriting)
        print(f"Ep {self.current_episode}/{self.max_episodes} ({progress_pct:.1f}%) | "
              f"R: {self.episode_rewards[-1]:.2f} | "
              f"Avg: {recent_avg:.2f} | "
              f"Turns: {ep_turns} | "
              f"{options_str} | "
              f"EpTime: {ep_time_str} | "
              f"Time {hours:02d}:{minutes:02d}:{seconds:02d} | "
              f"ETA: {eta_str} → {finish_str}", flush=True)
        
        # Every 50 episodes, print detailed progress
        if self.current_episode % 50 == 0:
            print()
            self._print_detailed_progress()
    
    def _print_detailed_progress(self):
        """Print detailed stats every 50 episodes"""
        print("\n" + "=" * 80)
        print(f"CHECKPOINT @ Episode {self.current_episode}/{self.max_episodes}")
        print("=" * 80)
        
        # Reward stats
        recent_100 = self.episode_rewards[-100:] if len(self.episode_rewards) >= 100 else self.episode_rewards
        print(f"Rewards:")
        print(f"  Last episode: {self.episode_rewards[-1]:.3f}")
        print(f"  Recent avg:   {sum(recent_100)/len(recent_100):.3f} (last {len(recent_100)} eps)")
        print(f"  Overall avg:  {sum(self.episode_rewards)/len(self.episode_rewards):.3f}")
        
        # Episode length
        recent_len = self.episode_lengths[-100:] if len(self.episode_lengths) >= 100 else self.episode_lengths
        print(f"\nEpisode Length:")
        print(f"  Last: {self.episode_lengths[-1]} turns")
        print(f"  Avg:  {sum(recent_len)/len(recent_len):.1f} turns")
        
        # Option distribution
        total_turns = sum(self.option_counts.values())
        print(f"\nOption Distribution (all episodes):")
        for option, count in sorted(self.option_counts.items(), key=lambda x: -x[1]):
            pct = (count / total_turns * 100) if total_turns > 0 else 0
            bar_len = int(pct / 2)  # Scale to 50 chars max
            bar = "#" * bar_len
            print(f"  {option:15s}: {bar} {pct:5.1f}% ({count})")
        
        # Time estimate
        elapsed = (datetime.now() - self.start_time).total_seconds()
        eps_per_sec = self.current_episode / elapsed if elapsed > 0 else 0
        remaining_eps = self.max_episodes - self.current_episode
        eta_sec = remaining_eps / eps_per_sec if eps_per_sec > 0 else 0
        eta_hours = int(eta_sec // 3600)
        eta_mins = int((eta_sec % 3600) // 60)
        
        print(f"\nTime:")
        print(f"  Elapsed: {int(elapsed//3600):02d}:{int((elapsed%3600)//60):02d}:{int(elapsed%60):02d}")
        print(f"  ETA:     {eta_hours:02d}:{eta_mins:02d}:00 ({eps_per_sec:.2f} eps/sec)")
        
        print("=" * 80 + "\n")

