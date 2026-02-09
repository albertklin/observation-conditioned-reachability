"""Keyboard controller for runtime control of safety filter.

Provides keyboard shortcuts for:
- (space/p): Pause/unpause control loop
- (q): Quit/terminate
- (i): Immobilize (force zero velocity)

Usage:
    from utils.keyboard_controller import KeyboardController

    controller = KeyboardController()
    controller.start()

    while not controller.should_terminate:
        if controller.is_paused:
            continue
        if controller.is_immobilized:
            v_x, v_yaw = 0, 0
        # ... rest of control loop

    controller.stop()
"""

import threading
from dataclasses import dataclass, field
from typing import Optional

try:
    from pynput import keyboard
    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False


# Terminal escape codes
CLR = '\x1B[0K'  # Clear to end of line


@dataclass
class KeyboardController:
    """Thread-safe keyboard controller for runtime control.

    Attributes:
        is_paused: If True, control loop should skip processing.
        should_terminate: If True, control loop should exit.
        is_immobilized: If True, force zero velocity output.
    """

    is_paused: bool = True  # Start paused for safety
    should_terminate: bool = False
    is_immobilized: bool = False

    _listener: Optional[object] = field(default=None, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self):
        if not PYNPUT_AVAILABLE:
            print("Warning: pynput not installed. Keyboard control disabled.")
            print("Install with: pip install pynput")

    def start(self):
        """Start listening for keyboard input."""
        if not PYNPUT_AVAILABLE:
            return

        self._listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )
        self._listener.start()
        self._print_options()

    def stop(self):
        """Stop listening for keyboard input."""
        if self._listener is not None:
            self._listener.stop()
            self._listener = None

    def _print_options(self):
        """Print available keyboard options."""
        print(f"Keyboard controls: (space/p)ause, (q)uit, (i)mmobilize{CLR}")

    def _on_press(self, key):
        """Handle key press events."""
        with self._lock:
            # Pause toggle (space or 'p')
            if key == keyboard.Key.space or self._is_char(key, 'p'):
                self.is_paused = not self.is_paused
                status = "PAUSED" if self.is_paused else "RUNNING"
                print(f"\r[{status}]{CLR}")

            # Quit ('q')
            elif self._is_char(key, 'q'):
                self.should_terminate = True
                print(f"\rTerminating...{CLR}")

            # Immobilize toggle ('i')
            elif self._is_char(key, 'i'):
                self.is_immobilized = not self.is_immobilized
                status = "IMMOBILIZED" if self.is_immobilized else "MOBILE"
                print(f"\r[{status}]{CLR}")

    def _on_release(self, key):
        """Handle key release events."""
        pass

    @staticmethod
    def _is_char(key, char: str) -> bool:
        """Check if key matches a character."""
        try:
            return key.char == char
        except AttributeError:
            return False


class DummyKeyboardController:
    """Dummy controller when pynput is not available or not needed."""

    is_paused: bool = False
    should_terminate: bool = False
    is_immobilized: bool = False

    def start(self):
        pass

    def stop(self):
        pass


def create_keyboard_controller(enabled: bool = True) -> KeyboardController:
    """Factory function to create appropriate keyboard controller.

    Args:
        enabled: If False, returns a dummy controller that does nothing.

    Returns:
        KeyboardController or DummyKeyboardController instance.
    """
    if enabled and PYNPUT_AVAILABLE:
        return KeyboardController()
    else:
        return DummyKeyboardController()
