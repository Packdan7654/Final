"""
Museum Floor Map Visualizer

Real-time 2D map showing:
- Exhibit locations (spatial layout)
- Agent position (where agent is directing attention)
- Visitor position (where visitor is looking)
- Engagement indicators (color-coded)
- Movement trails

Usage:
    map_viz = MuseumMapVisualizer(enabled=True, exhibits=exhibit_names)
    map_viz.update(agent_exhibit, visitor_exhibit, dwell, turn_num)
    map_viz.save_snapshot(f"maps/turn_{turn}.png")
    map_viz.save_animation("maps/episode_animation.gif")
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import deque


class MuseumMapVisualizer:
    """
    Real-time 2D museum floor map visualization.
    
    Shows agent and visitor positions relative to exhibits with engagement indicators.
    """
    
    # Museum layout: exhibit positions (x, y) in 2D space
    # Format: exhibit_name -> (x, y, width, height)
    # Layout for 5 exhibits arranged in museum floor
    EXHIBIT_LAYOUT = {
        "King_Caspar": (1, 4, 2.5, 1.5),      # Top left
        "Turban": (5, 4, 2.5, 1.5),           # Top center
        "Dom_Miguel": (9, 4, 2.5, 1.5),       # Top right
        "Pedro_Sunda": (3, 1, 2.5, 1.5),      # Bottom left
        "Diego_Bemba": (7, 1, 2.5, 1.5)       # Bottom right
    }
    
    # Colors
    COLOR_AGENT = '#2E86AB'  # Blue
    COLOR_VISITOR_HIGH = '#78BC61'  # Green (high engagement)
    COLOR_VISITOR_MED = '#F4D35E'  # Yellow (medium engagement)
    COLOR_VISITOR_LOW = '#DD6E42'  # Orange/Red (low engagement)
    COLOR_EXHIBIT_ACTIVE = '#FFBE0B'  # Bright yellow (current focus)
    COLOR_EXHIBIT_VISITED = '#B8B8B8'  # Gray (visited)
    COLOR_EXHIBIT_UNVISITED = '#E8E8E8'  # Light gray (not visited)
    
    def __init__(self, enabled: bool = True, exhibits: List[str] = None, 
                 save_dir: str = "training_logs/maps", live_display: bool = True):
        self.enabled = enabled
        self.live_display = live_display  # New parameter to control live display
        self.exhibits = exhibits or list(self.EXHIBIT_LAYOUT.keys())
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Tracking
        self.agent_position = None
        self.visitor_position = None
        self.current_dwell = 0.0
        self.visited_exhibits = set()
        
        # Completion rates: exhibit_name -> completion (0.0 to 1.0)
        self.exhibit_completion = {}
        
        # History for trails
        self.agent_trail = deque(maxlen=10)
        self.visitor_trail = deque(maxlen=10)
        
        # Frame storage for animation
        self.frames = []
        self.max_frames = 200  # Limit to prevent memory issues
        
        # Initialize plot if enabled
        if self.enabled:
            self._init_plot()
    
    def _init_plot(self):
        """Initialize the matplotlib figure and axes"""
        # Only enable interactive mode and show plot if live_display is True
        if self.live_display:
            plt.ion()
        
        self.fig, self.ax = plt.subplots(figsize=(14, 7))
        self.ax.set_xlim(0, 13)
        self.ax.set_ylim(0, 7)
        self.ax.set_aspect('equal')
        self.ax.set_facecolor('#F5F5F5')
        
        # Remove axis ticks
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        
        # Title
        self.ax.set_title('Museum Floor Map - Agent & Visitor Positions', 
                         fontsize=16, fontweight='bold', pad=20)
        
        # Add legend
        self._add_legend()
        
        # Only show the window if live_display is enabled
        if self.live_display:
            plt.show(block=False)
            plt.pause(0.001)  # Small pause to ensure window appears
    
    def _add_legend(self):
        """Add legend explaining symbols"""
        # Create sample patches for completion gradient
        completion_0, _ = self._get_completion_color(0.0)
        completion_50, _ = self._get_completion_color(0.5)
        completion_100, _ = self._get_completion_color(1.0)
        
        legend_elements = [
            patches.Patch(facecolor=self.COLOR_AGENT, label='Agent'),
            patches.Patch(facecolor=self.COLOR_VISITOR_HIGH, label='Visitor (High Engagement)'),
            patches.Patch(facecolor=self.COLOR_VISITOR_MED, label='Visitor (Med Engagement)'),
            patches.Patch(facecolor=self.COLOR_VISITOR_LOW, label='Visitor (Low Engagement)'),
            patches.Patch(facecolor=self.COLOR_EXHIBIT_ACTIVE, label='Current Focus'),
            patches.Patch(facecolor=completion_0, label='Exhibit: 0% Complete'),
            patches.Patch(facecolor=completion_50, label='Exhibit: 50% Complete'),
            patches.Patch(facecolor=completion_100, label='Exhibit: 100% Complete'),
            patches.Patch(facecolor='white', label='(Percentage shown in box)')
        ]
        self.ax.legend(handles=legend_elements, loc='upper left', 
                      bbox_to_anchor=(1.02, 1), fontsize=9)
    
    def _get_exhibit_center(self, exhibit_name: str) -> Tuple[float, float]:
        """Get center coordinates of an exhibit"""
        if exhibit_name not in self.EXHIBIT_LAYOUT:
            return (6.5, 3.5)  # Default center (middle of 5-exhibit layout)
        
        x, y, w, h = self.EXHIBIT_LAYOUT[exhibit_name]
        return (x + w/2, y + h/2)
    
    def _get_visitor_color(self, dwell: float) -> str:
        """Get visitor color based on engagement (dwell)"""
        if dwell > 0.7:
            return self.COLOR_VISITOR_HIGH
        elif dwell > 0.4:
            return self.COLOR_VISITOR_MED
        else:
            return self.COLOR_VISITOR_LOW
    
    def _get_completion_color(self, completion: float) -> Tuple[str, float]:
        """
        Get color for exhibit based on completion rate.
        Returns (color, alpha) tuple.
        Color gradient: light gray (0%) -> light green -> dark green (100%)
        """
        if completion is None or completion <= 0.0:
            # No completion - light gray
            return '#E8E8E8', 0.6
        elif completion >= 1.0:
            # Fully complete - dark green
            return '#2D5016', 0.9
        else:
            # Interpolate between light gray and dark green
            # Completion 0.0 -> #E8E8E8 (light gray)
            # Completion 0.5 -> #90C695 (medium green)
            # Completion 1.0 -> #2D5016 (dark green)
            
            # Use RGB interpolation
            # Light gray: RGB(232, 232, 232)
            # Medium green: RGB(144, 198, 149)
            # Dark green: RGB(45, 80, 22)
            
            if completion < 0.5:
                # Interpolate between light gray and medium green
                t = completion * 2  # Scale to 0-1
                r = int(232 + (144 - 232) * t)
                g = int(232 + (198 - 232) * t)
                b = int(232 + (149 - 232) * t)
            else:
                # Interpolate between medium green and dark green
                t = (completion - 0.5) * 2  # Scale to 0-1
                r = int(144 + (45 - 144) * t)
                g = int(198 + (80 - 198) * t)
                b = int(149 + (22 - 149) * t)
            
            # Alpha increases with completion
            alpha = 0.6 + (completion * 0.3)  # 0.6 to 0.9
            
            color = f'#{r:02x}{g:02x}{b:02x}'
            return color, alpha
    
    def _draw_exhibits(self):
        """Draw all exhibit rectangles with completion-based coloring"""
        for exhibit_name, (x, y, w, h) in self.EXHIBIT_LAYOUT.items():
            # Get completion rate (default to 0 if not provided)
            completion = self.exhibit_completion.get(exhibit_name, 0.0)
            
            # Get color based on completion
            completion_color, completion_alpha = self._get_completion_color(completion)
            
            # Determine border/style based on status
            if exhibit_name == self.agent_position or exhibit_name == self.visitor_position:
                # Currently active - highlight with bright border
                border_color = self.COLOR_EXHIBIT_ACTIVE
                linewidth = 3
                linestyle = '-'
                # Use completion color but brighter when active
                face_color = completion_color
                face_alpha = min(1.0, completion_alpha + 0.2)
            elif exhibit_name in self.visited_exhibits:
                # Visited - use completion color
                border_color = 'black'
                linewidth = 2
                linestyle = '--'
                face_color = completion_color
                face_alpha = completion_alpha
            else:
                # Not visited - light gray with completion hint
                border_color = 'gray'
                linewidth = 1
                linestyle = ':'
                face_color = completion_color
                face_alpha = completion_alpha * 0.7  # Slightly more transparent
            
            # Draw exhibit rectangle
            rect = patches.Rectangle((x, y), w, h, 
                                     linewidth=linewidth, 
                                     edgecolor=border_color,
                                     facecolor=face_color,
                                     linestyle=linestyle,
                                     alpha=face_alpha)
            self.ax.add_patch(rect)
            
            # Add exhibit name (abbreviated) at top
            name_short = exhibit_name.replace('_', ' ')
            if len(name_short) > 20:
                name_short = name_short[:17] + "..."
            
            # Text color: white if completion is high, black otherwise
            text_color = 'white' if completion > 0.5 else '#333333'
            
            self.ax.text(x + w/2, y + h - 0.25, name_short,
                        ha='center', va='top',
                        fontsize=8, fontweight='bold',
                        color=text_color,
                        bbox=dict(boxstyle='round,pad=0.2', 
                                 facecolor='white' if completion <= 0.5 else 'black', 
                                 alpha=0.8 if completion <= 0.5 else 0.6,
                                 edgecolor='none'))
            
            # Add completion percentage in center
            completion_pct = completion * 100
            completion_text = f"{completion_pct:.0f}%"
            
            # Use contrasting text color
            center_text_color = 'white' if completion > 0.4 else '#333333'
            
            self.ax.text(x + w/2, y + h/2, completion_text,
                        ha='center', va='center',
                        fontsize=10, fontweight='bold',
                        color=center_text_color,
                        bbox=dict(boxstyle='round,pad=0.3', 
                                 facecolor='white' if completion <= 0.3 else 'black', 
                                 alpha=0.9,
                                 edgecolor='none'))
    
    def _draw_agent(self):
        """Draw agent position"""
        if self.agent_position:
            cx, cy = self._get_exhibit_center(self.agent_position)
            
            # Draw agent marker
            agent_marker = patches.Circle((cx - 0.5, cy + 0.8), 0.3,
                                         facecolor=self.COLOR_AGENT,
                                         edgecolor='black',
                                         linewidth=2,
                                         zorder=10)
            self.ax.add_patch(agent_marker)
            
            # Add label (text instead of emoji for compatibility)
            self.ax.text(cx - 0.5, cy + 0.8, 'A',
                        ha='center', va='center',
                        fontsize=14, fontweight='bold', 
                        color='white', zorder=11)
            
            # Draw trail
            if len(self.agent_trail) > 1:
                trail_x = [self._get_exhibit_center(ex)[0] - 0.5 for ex in self.agent_trail]
                trail_y = [self._get_exhibit_center(ex)[1] + 0.8 for ex in self.agent_trail]
                self.ax.plot(trail_x, trail_y, 
                           color=self.COLOR_AGENT, 
                           alpha=0.3, 
                           linewidth=2, 
                           linestyle='--',
                           zorder=5)
    
    def _draw_visitor(self):
        """Draw visitor position with engagement indicator"""
        if self.visitor_position:
            cx, cy = self._get_exhibit_center(self.visitor_position)
            visitor_color = self._get_visitor_color(self.current_dwell)
            
            # Draw visitor marker
            visitor_marker = patches.Circle((cx + 0.5, cy + 0.8), 0.3,
                                           facecolor=visitor_color,
                                           edgecolor='black',
                                           linewidth=2,
                                           zorder=10)
            self.ax.add_patch(visitor_marker)
            
            # Add label (text instead of emoji for compatibility)
            self.ax.text(cx + 0.5, cy + 0.8, 'V',
                        ha='center', va='center',
                        fontsize=14, fontweight='bold',
                        color='white', zorder=11)
            
            # Draw engagement bar above visitor
            bar_width = 0.6
            bar_height = 0.1
            bar_x = cx + 0.5 - bar_width/2
            bar_y = cy + 1.2
            
            # Background bar
            bg_bar = patches.Rectangle((bar_x, bar_y), bar_width, bar_height,
                                       facecolor='white',
                                       edgecolor='black',
                                       linewidth=1,
                                       zorder=9)
            self.ax.add_patch(bg_bar)
            
            # Engagement fill
            fill_width = bar_width * self.current_dwell
            fill_bar = patches.Rectangle((bar_x, bar_y), fill_width, bar_height,
                                         facecolor=visitor_color,
                                         edgecolor='none',
                                         zorder=10)
            self.ax.add_patch(fill_bar)
            
            # Dwell text
            self.ax.text(cx + 0.5, bar_y + bar_height + 0.15, 
                        f'{self.current_dwell:.2f}',
                        ha='center', va='bottom',
                        fontsize=8, fontweight='bold',
                        color=visitor_color)
            
            # Draw trail
            if len(self.visitor_trail) > 1:
                trail_x = [self._get_exhibit_center(ex)[0] + 0.5 for ex in self.visitor_trail]
                trail_y = [self._get_exhibit_center(ex)[1] + 0.8 for ex in self.visitor_trail]
                self.ax.plot(trail_x, trail_y, 
                           color=visitor_color, 
                           alpha=0.3, 
                           linewidth=2, 
                           linestyle='--',
                           zorder=5)
    
    def _draw_connection(self):
        """Draw line connecting agent and visitor if at different exhibits"""
        if self.agent_position and self.visitor_position:
            if self.agent_position != self.visitor_position:
                # Different exhibits - show misalignment
                agent_cx, agent_cy = self._get_exhibit_center(self.agent_position)
                visitor_cx, visitor_cy = self._get_exhibit_center(self.visitor_position)
                
                self.ax.plot([agent_cx - 0.5, visitor_cx + 0.5],
                           [agent_cy + 0.8, visitor_cy + 0.8],
                           color='red', linewidth=2, linestyle=':', 
                           alpha=0.5, zorder=4)
                
                # Add warning icon (text instead of emoji)
                mid_x = (agent_cx - 0.5 + visitor_cx + 0.5) / 2
                mid_y = (agent_cy + 0.8 + visitor_cy + 0.8) / 2
                self.ax.text(mid_x, mid_y, '!',
                           ha='center', va='center',
                           fontsize=16, fontweight='bold',
                           color='red', zorder=11,
                           bbox=dict(boxstyle='round,pad=0.3',
                                    facecolor='white',
                                    edgecolor='red',
                                    linewidth=2,
                                    alpha=0.9))
            else:
                # Same exhibit - show alignment
                cx, cy = self._get_exhibit_center(self.agent_position)
                
                # Draw alignment indicator
                self.ax.plot([cx - 0.5, cx + 0.5],
                           [cy + 0.8, cy + 0.8],
                           color='green', linewidth=3, 
                           alpha=0.5, zorder=4)
    
    def update(self, agent_exhibit: str, visitor_exhibit: str, 
               dwell: float, turn_num: int = 0,
               option: str = None,
               exhibit_completion: Dict[str, float] = None):
        """
        Update the map with new positions.
        
        Args:
            agent_exhibit: Where agent is directing attention
            visitor_exhibit: Where visitor is looking
            dwell: Current engagement (0-1)
            turn_num: Current turn number
            option: Current agent option (for annotation)
            exhibit_completion: Dict mapping exhibit names to completion rates (0.0 to 1.0)
        """
        if not self.enabled:
            return
        
        # Update state
        self.agent_position = agent_exhibit
        self.visitor_position = visitor_exhibit
        self.current_dwell = dwell
        
        # Update completion rates if provided
        if exhibit_completion is not None:
            self.exhibit_completion.update(exhibit_completion)
        
        # Track visited exhibits
        if agent_exhibit:
            self.visited_exhibits.add(agent_exhibit)
        if visitor_exhibit:
            self.visited_exhibits.add(visitor_exhibit)
        
        # Update trails
        if agent_exhibit and (not self.agent_trail or self.agent_trail[-1] != agent_exhibit):
            self.agent_trail.append(agent_exhibit)
        if visitor_exhibit and (not self.visitor_trail or self.visitor_trail[-1] != visitor_exhibit):
            self.visitor_trail.append(visitor_exhibit)
        
        # Clear and redraw
        self.ax.clear()
        self.ax.set_xlim(0, 13)
        self.ax.set_ylim(0, 7)
        self.ax.set_aspect('equal')
        self.ax.set_facecolor('#F5F5F5')
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        
        # Draw components
        self._draw_exhibits()
        self._draw_connection()
        self._draw_agent()
        self._draw_visitor()
        self._add_legend()
        
        # Add turn info
        alignment = "[OK] ALIGNED" if agent_exhibit == visitor_exhibit else "[!] MISALIGNED"
        alignment_color = "green" if agent_exhibit == visitor_exhibit else "red"
        
        info_text = f"Turn {turn_num}"
        if option:
            info_text += f" | Option: {option}"
        info_text += f" | {alignment}"
        
        self.ax.text(0.5, 0.98, info_text,
                    transform=self.ax.transAxes,
                    ha='center', va='top',
                    fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5',
                             facecolor='white',
                             edgecolor=alignment_color,
                             linewidth=2))
        
        # Tight layout
        plt.tight_layout()
        
        # Only refresh display if live_display is enabled
        if self.live_display:
            plt.pause(0.001)
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
    
    def save_snapshot(self, filename: str):
        """Save current state as PNG"""
        if not self.enabled:
            return
        
        filepath = self.save_dir / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        self.fig.savefig(filepath, dpi=150, bbox_inches='tight')
    
    def capture_frame(self):
        """Capture current state for animation"""
        if not self.enabled or len(self.frames) >= self.max_frames:
            return
        
        # Save current figure as array (use most reliable method)
        self.fig.canvas.draw()
        
        # Use PIL-based approach for best compatibility
        try:
            import io
            from PIL import Image
            
            # Save to buffer and read back
            buf = io.BytesIO()
            self.fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            img = Image.open(buf)
            frame = np.array(img)
            
            # Ensure RGB (convert RGBA if needed)
            if frame.shape[2] == 4:
                frame = frame[:, :, :3]
            
            buf.close()
            self.frames.append(frame)
        except Exception as e:
            # Silently skip frame if capture fails
            print(f"Warning: Failed to capture frame: {e}")
    
    def save_animation(self, filename: str = "episode_animation.gif", fps: int = 2):
        """
        Save accumulated frames as animated GIF.
        
        Args:
            filename: Output filename
            fps: Frames per second (default 2 = 0.5s per frame)
        """
        if not self.enabled or not self.frames:
            print("⚠️  No frames to animate")
            return
        
        filepath = self.save_dir / filename
        
        # Create animation
        fig_anim, ax_anim = plt.subplots(figsize=(16, 8))
        ax_anim.axis('off')
        
        im = ax_anim.imshow(self.frames[0])
        
        def update_frame(frame_idx):
            im.set_array(self.frames[frame_idx])
            return [im]
        
        anim = FuncAnimation(fig_anim, update_frame, 
                            frames=len(self.frames),
                            interval=1000/fps,
                            blit=True)
        
        # Save as GIF
        writer = PillowWriter(fps=fps)
        anim.save(filepath, writer=writer)
        
        plt.close(fig_anim)
        print(f"✓ Saved animation to {filepath} ({len(self.frames)} frames)")
    
    def reset(self):
        """Reset for new episode"""
        self.agent_position = None
        self.visitor_position = None
        self.current_dwell = 0.0
        self.visited_exhibits.clear()
        self.exhibit_completion.clear()
        self.agent_trail.clear()
        self.visitor_trail.clear()
        self.frames.clear()
    
    def show(self):
        """Display the current map (blocking)"""
        if self.enabled:
            plt.show()
    
    def close(self):
        """Close the figure and cleanup"""
        if self.enabled and hasattr(self, 'fig'):
            plt.close(self.fig)
            if self.live_display:
                plt.ioff()  # Only turn off interactive mode if it was turned on


def test_visualizer():
    """Test the visualizer with sample data"""
    import time
    
    exhibits = list(MuseumMapVisualizer.EXHIBIT_LAYOUT.keys())
    
    # Test with live_display=False (saves files but doesn't show windows)
    viz = MuseumMapVisualizer(enabled=True, live_display=False)
    
    # Simulate a tour
    tour = [
        (exhibits[0], exhibits[0], 0.8, "Explain"),  # Aligned, high engagement
        (exhibits[0], exhibits[0], 0.9, "Explain"),
        (exhibits[0], exhibits[0], 0.6, "AskQuestion"),
        (exhibits[1], exhibits[0], 0.4, "OfferTransition"),  # Misaligned!
        (exhibits[1], exhibits[1], 0.7, "Explain"),  # Aligned again
        (exhibits[1], exhibits[1], 0.8, "Explain"),
        (exhibits[2], exhibits[2], 0.75, "Explain"),
        (exhibits[3], exhibits[3], 0.85, "Explain"),
    ]
    
    for i, (agent_ex, visitor_ex, dwell, option) in enumerate(tour):
        viz.update(agent_ex, visitor_ex, dwell, turn_num=i+1, option=option)
        viz.capture_frame()
        viz.save_snapshot(f"test_turn_{i+1:02d}.png")
        time.sleep(0.1)  # Reduced since no live display
    
    # Save animation
    viz.save_animation("test_episode.gif", fps=1)
    
    viz.close()
    
    print("✓ Test complete! Check training_logs/maps/ for outputs")
    print("✓ No live windows were displayed - files saved only")


if __name__ == "__main__":
    test_visualizer()

