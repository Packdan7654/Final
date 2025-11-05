# src/visualization/live_training_monitor.py

"""
Live Training Monitor for HRL Museum Agent

Provides real-time visualization of training with:
- Turn-by-turn dialogue display
- Gaze/engagement visualization
- Exhibit coverage progress
- Reward decomposition
- Option timeline

Usage:
    monitor = LiveTrainingMonitor(enabled=True)
    monitor.on_episode_start(episode_num, persona)
    monitor.on_turn(turn_data)
    monitor.on_episode_end(episode_summary)
"""

import time
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
from pathlib import Path

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.live import Live
    from rich.progress import Progress, BarColumn, TextColumn
    from rich.text import Text
    from rich.columns import Columns
    from rich import box
    _has_rich = True
except ImportError:
    _has_rich = False
    print("[WARNING] rich library not installed. Install with: pip install rich")

import numpy as np


class LiveTrainingMonitor:
    """
    Real-time training visualization with turn-by-turn dialogue tracking.
    
    Features:
    - Color-coded dialogue (agent in blue, visitor in yellow)
    - Live gaze bar chart (dwell time visualization)
    - Exhibit coverage progress bars
    - Reward component breakdown
    - Option usage timeline
    - Cumulative metrics
    """
    
    def __init__(self, enabled: bool = True, log_dir: str = "training_logs"):
        self.enabled = enabled and _has_rich
        self.console = Console() if self.enabled else None
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Episode state
        self.episode_num = 0
        self.persona = ""
        self.turn_count = 0
        self.dialogue_history = []
        self.exhibit_coverage = {}
        self.cumulative_reward = 0.0
        self.reward_components = {
            "engagement": 0.0,
            "novelty": 0.0,
            "responsiveness": 0.0,
            "transition": 0.0,
            "conclude": 0.0
        }
        self.option_timeline = []
        
        # Metrics accumulation
        self.episode_metrics = []
        self.turn_metrics = []
        
    def on_episode_start(self, episode_num: int, persona: str, exhibits: List[str]):
        """Called at the start of each episode"""
        if not self.enabled:
            return
            
        self.episode_num = episode_num
        self.persona = persona
        self.turn_count = 0
        self.dialogue_history = []
        self.exhibit_coverage = {ex: {"total": 0, "mentioned": 0} for ex in exhibits}
        self.cumulative_reward = 0.0
        self.reward_components = {k: 0.0 for k in self.reward_components}
        self.option_timeline = []
        
        # Print episode header
        self.console.clear()
        self.console.rule(f"[bold cyan]Episode {episode_num} - Persona: {persona}[/bold cyan]", style="cyan")
        print()
        
    def on_turn(self, turn_data: Dict[str, Any]):
        """Called after each turn with comprehensive turn data"""
        if not self.enabled:
            return
            
        self.turn_count += 1
        
        # Extract data
        agent_utt = turn_data.get("agent_utterance", "")
        user_utt = turn_data.get("user_utterance", "")
        option = turn_data.get("option", "")
        subaction = turn_data.get("subaction", "")
        dwell = turn_data.get("dwell", 0.0)
        response_type = turn_data.get("response_type", "")
        
        # Rewards
        r_eng = turn_data.get("reward_engagement", 0.0)
        r_nov = turn_data.get("reward_novelty", 0.0)
        r_resp = turn_data.get("reward_responsiveness", 0.0)
        r_trans = turn_data.get("reward_transition_penalty", 0.0)
        r_conclude = turn_data.get("reward_conclude", 0.0)
        total_reward = turn_data.get("total_reward", 0.0)
        
        # Exhibit data
        current_exhibit = turn_data.get("current_exhibit", "Unknown")
        facts_shared = turn_data.get("facts_shared", 0)
        
        # Update tracking
        self.cumulative_reward += total_reward
        self.reward_components["engagement"] += r_eng
        self.reward_components["novelty"] += r_nov
        self.reward_components["responsiveness"] += r_resp
        self.reward_components["transition"] += r_trans
        self.reward_components["conclude"] += r_conclude
        self.option_timeline.append(option)
        
        # Update exhibit coverage if provided
        if "exhibit_coverage" in turn_data:
            self.exhibit_coverage = turn_data["exhibit_coverage"]
        
        # Store for history
        self.dialogue_history.append({
            "turn": self.turn_count,
            "agent": agent_utt,
            "user": user_utt,
            "option": option,
            "subaction": subaction,
            "dwell": dwell,
            "response_type": response_type,
            "reward": total_reward
        })
        
        # === RENDER TURN ===
        self._render_turn(
            agent_utt=agent_utt,
            user_utt=user_utt,
            option=option,
            subaction=subaction,
            dwell=dwell,
            response_type=response_type,
            current_exhibit=current_exhibit,
            facts_shared=facts_shared,
            r_eng=r_eng,
            r_nov=r_nov,
            r_resp=r_resp,
            r_trans=r_trans,
            r_conclude=r_conclude,
            total_reward=total_reward
        )
        
        # Store turn metrics
        self.turn_metrics.append({
            "episode": self.episode_num,
            "turn": self.turn_count,
            "option": option,
            "subaction": subaction,
            "dwell": dwell,
            "response_type": response_type,
            "reward": total_reward,
            "cumulative_reward": self.cumulative_reward
        })
        
    def _render_turn(self, agent_utt, user_utt, option, subaction, dwell, 
                     response_type, current_exhibit, facts_shared,
                     r_eng, r_nov, r_resp, r_trans, r_conclude, total_reward):
        """Render a single turn with all visualizations"""
        
        # === TURN HEADER ===
        self.console.print(f"\n[bold]═══ Turn {self.turn_count} ═══[/bold]", style="white")
        
        # === AGENT UTTERANCE ===
        agent_panel = Panel(
            Text(agent_utt, style="bold cyan"),
            title=f"[cyan]🤖 Agent[/cyan] ({option} → {subaction})",
            border_style="cyan",
            box=box.ROUNDED
        )
        self.console.print(agent_panel)
        
        # === VISITOR UTTERANCE ===
        # Color code by response type
        response_colors = {
            "question": "yellow",
            "acknowledgment": "green",
            "follow_up_question": "bright_green",
            "statement": "white",
            "confusion": "red",
            "silence": "dim"
        }
        user_color = response_colors.get(response_type, "white")
        
        user_panel = Panel(
            Text(user_utt or "[silence]", style=f"bold {user_color}"),
            title=f"[{user_color}]👤 Visitor[/{user_color}] ({response_type})",
            border_style=user_color,
            box=box.ROUNDED
        )
        self.console.print(user_panel)
        
        # === GAZE & ENGAGEMENT ===
        gaze_bar = self._create_bar(dwell, width=30, filled_char="█", empty_char="░")
        gaze_color = "green" if dwell > 0.7 else "yellow" if dwell > 0.4 else "red"
        self.console.print(f"  👁️  Dwell: [{gaze_color}]{gaze_bar}[/{gaze_color}] {dwell:.2f}")
        
        # === EXHIBIT STATUS ===
        self.console.print(f"  🎨 Exhibit: [bold]{current_exhibit}[/bold] | Facts shared: {facts_shared}")
        
        # === REWARD BREAKDOWN ===
        reward_table = Table(show_header=True, header_style="bold magenta", box=box.SIMPLE)
        reward_table.add_column("Component", style="dim")
        reward_table.add_column("Value", justify="right")
        reward_table.add_column("Visual", justify="left")
        
        reward_data = [
            ("Engagement", r_eng, "green"),
            ("Novelty", r_nov, "blue"),
            ("Responsiveness", r_resp, "cyan" if r_resp > 0 else "red" if r_resp < 0 else "dim"),
            ("Transition", r_trans, "red" if r_trans < 0 else "dim"),
            ("Conclude", r_conclude, "yellow" if r_conclude > 0 else "dim"),
            ("TOTAL", total_reward, "bold magenta")
        ]
        
        for name, value, color in reward_data:
            visual = self._create_reward_bar(value)
            reward_table.add_row(
                name,
                f"[{color}]{value:+.3f}[/{color}]",
                f"[{color}]{visual}[/{color}]"
            )
        
        self.console.print(reward_table)
        
        # === CUMULATIVE METRICS ===
        self.console.print(
            f"  📊 Cumulative Reward: [bold magenta]{self.cumulative_reward:.2f}[/bold magenta] | "
            f"Turn: {self.turn_count}"
        )
        
    def _create_bar(self, value: float, width: int = 20, 
                   filled_char: str = "█", empty_char: str = "░") -> str:
        """Create a visual bar for a 0-1 value"""
        filled = int(value * width)
        empty = width - filled
        return filled_char * filled + empty_char * empty
        
    def _create_reward_bar(self, value: float, scale: float = 0.5) -> str:
        """Create a visual bar for reward values (can be negative)"""
        if value == 0:
            return "·"
        
        length = int(abs(value) / scale * 10)
        length = min(length, 20)  # Cap at 20 chars
        
        if value > 0:
            return "▶" + "█" * length
        else:
            return "◀" + "█" * length
    
    def on_episode_end(self, episode_summary: Dict[str, Any]):
        """Called at episode end with summary statistics"""
        if not self.enabled:
            return
            
        # === EPISODE SUMMARY ===
        self.console.print("\n")
        self.console.rule("[bold green]Episode Complete[/bold green]", style="green")
        
        # Create summary table
        summary_table = Table(show_header=True, header_style="bold cyan", box=box.ROUNDED)
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", justify="right", style="yellow")
        
        # Add metrics
        summary_table.add_row("Total Turns", str(self.turn_count))
        summary_table.add_row("Cumulative Reward", f"{self.cumulative_reward:.2f}")
        summary_table.add_row("Avg Reward/Turn", f"{self.cumulative_reward / max(self.turn_count, 1):.3f}")
        
        # Reward breakdown
        summary_table.add_row("", "")  # Spacer
        for name, value in self.reward_components.items():
            summary_table.add_row(f"  {name.capitalize()}", f"{value:.2f}")
        
        # Exhibit coverage
        exhibits_covered = sum(1 for ex_data in self.exhibit_coverage.values() 
                              if ex_data.get("mentioned", 0) > 0)
        summary_table.add_row("", "")  # Spacer
        summary_table.add_row("Exhibits Covered", f"{exhibits_covered}/{len(self.exhibit_coverage)}")
        
        # Option usage
        option_counts = {}
        for opt in self.option_timeline:
            option_counts[opt] = option_counts.get(opt, 0) + 1
        
        summary_table.add_row("", "")  # Spacer
        for opt, count in sorted(option_counts.items(), key=lambda x: -x[1]):
            summary_table.add_row(f"  {opt}", str(count))
        
        self.console.print(summary_table)
        
        # === EXHIBIT COVERAGE VISUALIZATION ===
        if self.exhibit_coverage:
            self.console.print("\n[bold cyan]Exhibit Coverage:[/bold cyan]")
            for exhibit_name, data in self.exhibit_coverage.items():
                total = data.get("total", 1)
                mentioned = data.get("mentioned", 0)
                coverage = mentioned / total if total > 0 else 0.0
                
                bar = self._create_bar(coverage, width=25)
                color = "green" if coverage > 0.6 else "yellow" if coverage > 0.3 else "red"
                
                self.console.print(
                    f"  {exhibit_name[:30]:<30} [{color}]{bar}[/{color}] "
                    f"{mentioned}/{total} ({coverage*100:.0f}%)"
                )
        
        # === OPTION TIMELINE ===
        self.console.print("\n[bold cyan]Option Timeline:[/bold cyan]")
        timeline_str = ""
        for i, opt in enumerate(self.option_timeline):
            opt_symbol = self._get_option_symbol(opt)
            timeline_str += opt_symbol
            if (i + 1) % 40 == 0:
                timeline_str += "\n"
        self.console.print(f"  {timeline_str}")
        self.console.print("  E=Explain, A=Ask, T=Transition, C=Conclude")
        
        # Store episode metrics
        self.episode_metrics.append({
            "episode": self.episode_num,
            "persona": self.persona,
            "turns": self.turn_count,
            "cumulative_reward": self.cumulative_reward,
            "exhibits_covered": exhibits_covered,
            "option_counts": option_counts,
            **self.reward_components
        })
        
        # Wait for user to see summary
        time.sleep(0.5)
        
    def _get_option_symbol(self, option: str) -> str:
        """Get a single-character symbol for an option"""
        symbols = {
            "Explain": "[cyan]E[/cyan]",
            "AskQuestion": "[yellow]A[/yellow]",
            "OfferTransition": "[magenta]T[/magenta]",
            "Conclude": "[green]C[/green]"
        }
        return symbols.get(option, "?")
    
    def save_metrics(self, filename_prefix: str = "metrics"):
        """Save accumulated metrics to JSON files (only if there's data)"""
        if not self.episode_metrics and not self.turn_metrics:
            return  # Skip saving if no data
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save episode metrics (only if there's data)
        if self.episode_metrics:
            episode_file = self.log_dir / f"{filename_prefix}_episodes_{timestamp}.json"
            with open(episode_file, 'w') as f:
                json.dump(self.episode_metrics, f, indent=2)
        
        # Save turn metrics (only if there's data)
        if self.turn_metrics:
            turn_file = self.log_dir / f"{filename_prefix}_turns_{timestamp}.json"
            with open(turn_file, 'w') as f:
                json.dump(self.turn_metrics, f, indent=2)
        
        print(f"[INFO] Metrics saved to {self.log_dir}")
        
    def get_episode_metrics(self) -> List[Dict]:
        """Get accumulated episode metrics for analysis"""
        return self.episode_metrics
        
    def get_turn_metrics(self) -> List[Dict]:
        """Get accumulated turn metrics for analysis"""
        return self.turn_metrics