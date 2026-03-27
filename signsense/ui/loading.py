"""
ui/loading.py
=============
Loading screen component for SignSense.

Displays an animated progress bar with percentage and status messages
while the play mode initializes, providing visual feedback to prevent
UI freeze perception.

Visual Design
-------------
- Dark background matching app theme
- Centered animated progress bar with rounded corners
- Percentage text above the bar
- Status message below the bar
- Pulsing accent element for visual interest
- App title displayed at top
"""

import cv2
import numpy as np
import math
import time
from typing import Optional

# ---------------------------------------------------------------------------
# Palette - matching menu.py theme
# ---------------------------------------------------------------------------
BG          = (20, 20, 25)
PANEL_BG    = (35, 38, 45)
ACCENT      = (70, 130, 220)
ACCENT2     = (100, 100, 120)
WHITE       = (245, 245, 250)
TEXT_DIM    = (130, 135, 145)
DIM         = (120, 125, 135)
GREEN       = (80, 180, 120)

FONT        = cv2.FONT_HERSHEY_SIMPLEX
FONT_MONO   = cv2.FONT_HERSHEY_PLAIN


class LoadingScreen:
    """
    Displays a loading screen with animated progress bar and status messages.
    
    Usage:
        loading = LoadingScreen(W, H)
        cv2.namedWindow("SignSense")
        
        # Update progress during loading
        loading.update_progress(25, "Loading hand tracker...")
        cv2.imshow("SignSense", loading.render())
        
        # ... more loading steps ...
        
        # Final frame
        cv2.imshow("SignSense", loading.render())
    """
    
    def __init__(self, W: int = 640, H: int = 480):
        self.W = W
        self.H = H
        self._t0 = time.time()
        self._progress = 0.0
        self._status = "Initializing..."
        self._target_progress = 0.0
        self._target_status = "Initializing..."
        
    def update_progress(self, percent: float, status: str = None):
        """
        Update the loading progress.
        
        Args:
            percent: Progress percentage (0-100)
            status: Optional status message to display
        """
        self._target_progress = max(0.0, min(100.0, percent))
        if status is not None:
            self._target_status = status
            
    def _smooth_progress(self):
        """Smoothly interpolate progress for animation effect."""
        if self._progress < self._target_progress:
            # Smooth approach to target
            diff = self._target_progress - self._progress
            self._progress += diff * 0.15  # Smooth animation
            if abs(diff) < 0.5:
                self._progress = self._target_progress
        self._progress = max(0.0, min(100.0, self._progress))
        
    def reset(self):
        """Reset the loading screen for reuse."""
        self._progress = 0.0
        self._status = "Initializing..."
        self._target_progress = 0.0
        self._target_status = "Initializing..."
        self._t0 = time.time()
        
    def render(self) -> np.ndarray:
        """
        Render the loading screen frame.
        
        Returns:
            BGR frame ready for cv2.imshow()
        """
        # Smooth animation
        self._smooth_progress()
        self._status = self._target_status
        
        frame = np.zeros((self.H, self.W, 3), dtype=np.uint8)
        frame[:] = BG
        
        # Time for animations
        t = time.time() - self._t0
        
        # === Title ===
        cv2.putText(frame, "SignSense",
                   (self.W // 2 - 100, 80), FONT, 1.2, WHITE, 2, cv2.LINE_AA)
        
        # Subtitle
        cv2.putText(frame, "Loading...",
                   (self.W // 2 - 50, 115), FONT, 0.5, TEXT_DIM, 1, cv2.LINE_AA)
        
        # === Progress Bar Background ===
        bar_w = 300
        bar_h = 24
        bar_x = (self.W - bar_w) // 2
        bar_y = self.H // 2 - 20
        
        # Draw background panel
        cv2.rectangle(frame, 
                     (bar_x - 20, bar_y - 40), 
                     (bar_x + bar_w + 20, bar_y + bar_h + 60),
                     PANEL_BG, -1)
        cv2.rectangle(frame,
                     (bar_x - 20, bar_y - 40),
                     (bar_x + bar_w + 20, bar_y + bar_h + 60),
                     ACCENT2, 1)
        
        # === Percentage Text ===
        pct_text = f"{int(self._progress)}%"
        (pct_w, pct_h), _ = cv2.getTextSize(pct_text, FONT, 0.7, 2)
        cv2.putText(frame, pct_text,
                   (self.W // 2 - pct_w // 2, bar_y - 15),
                   FONT, 0.7, WHITE, 2, cv2.LINE_AA)
        
        # === Progress Bar Fill ===
        fill_width = int((self._progress / 100.0) * bar_w)
        if fill_width > 0:
            # Animated glow effect on the progress bar
            pulse = math.sin(t * 3) * 0.15 + 0.85
            glow_color = tuple(int(c * pulse) for c in ACCENT)
            
            # Draw filled portion with rounded corners
            r = 6  # corner radius
            # Main fill
            cv2.rectangle(frame, 
                        (bar_x, bar_y + r), 
                        (bar_x + fill_width, bar_y + bar_h - r),
                        glow_color, -1)
            # Top rounded corners
            if fill_width > r:
                cv2.rectangle(frame,
                            (bar_x, bar_y),
                            (bar_x + fill_width - r, bar_y + r),
                            glow_color, -1)
                cv2.circle(frame, (bar_x + r, bar_y + r), r, glow_color, -1)
            else:
                cv2.circle(frame, (bar_x + fill_width, bar_y + r), fill_width, glow_color, -1)
            # Bottom rounded corners
            if fill_width > r:
                cv2.rectangle(frame,
                            (bar_x, bar_y + bar_h - r),
                            (bar_x + fill_width - r, bar_y + bar_h),
                            glow_color, -1)
                cv2.circle(frame, (bar_x + r, bar_y + bar_h - r), r, glow_color, -1)
            else:
                cv2.circle(frame, (bar_x + fill_width, bar_y + bar_h - r), fill_width, glow_color, -1)
                
        # === Progress Bar Border ===
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                     ACCENT2, 1)
        
        # === Status Text ===
        (status_w, status_h), _ = cv2.getTextSize(self._status, FONT, 0.5, 1)
        cv2.putText(frame, self._status,
                   (self.W // 2 - status_w // 2, bar_y + bar_h + 30),
                   FONT, 0.5, TEXT_DIM, 1, cv2.LINE_AA)
        
        # === Pulsing Indicator ===
        # Small animated dots below status
        num_dots = 3
        dot_spacing = 20
        dot_start = self.W // 2 - ((num_dots - 1) * dot_spacing) // 2
        dot_y = bar_y + bar_h + 55
        
        for i in range(num_dots):
            # Staggered pulse animation
            phase = t * 2 + i * 0.5
            intensity = int(80 + 40 * math.sin(phase))
            dot_color = (intensity, intensity + 30, intensity + 50)
            cv2.circle(frame, (dot_start + i * dot_spacing, dot_y), 4, dot_color, -1)
        
        # === Ready State ===
        if self._progress >= 99.5:
            # Show green ready indicator
            ready_text = "Ready!"
            (ready_w, ready_h), _ = cv2.getTextSize(ready_text, FONT, 0.6, 2)
            cv2.putText(frame, ready_text,
                       (self.W // 2 - ready_w // 2, bar_y + bar_h + 85),
                       FONT, 0.6, GREEN, 2, cv2.LINE_AA)
        
        return frame


def render_loading_frame(W: int = 640, H: int = 480, 
                         progress: float = 0.0, 
                         status: str = "Loading...") -> np.ndarray:
    """
    Helper function to render a single loading frame.
    
    Args:
        W: Window width
        H: Window height
        progress: Progress percentage (0-100)
        status: Status message
        
    Returns:
        BGR frame ready for display
    """
    screen = LoadingScreen(W, H)
    screen.update_progress(progress, status)
    return screen.render()