"""Moon Lander — classic Atari-style lunar lander RL environment.

Core (``moonlander.core``) is pure stdlib so it runs in the browser via
Pyodide with zero package downloads. The Gymnasium wrapper
(``moonlander.env``) needs gymnasium+numpy and is only imported on the
Python side — never import it at package top level; the registration below
references it lazily via an entry-point string (CONTRACT §7).
"""

__version__ = "0.2.0"

try:
    import gymnasium as _gymnasium
except ImportError:  # browser/Pyodide: no gymnasium — the core imports cleanly
    pass
else:
    _gymnasium.register(
        id="MoonLander-v0",
        entry_point="moonlander.env:MoonLanderEnv",
    )
