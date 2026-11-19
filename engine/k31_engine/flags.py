"""Runtime flags: K31_* settings that may change while the server runs, so one warm server can measure several
configurations of the prompt path without a restart.

The control file (K31_FLAGS_FILE, default /kl/k31-flags.json, a JSON object {"K31_NAME": value, ...}) is re-read
between steps when its modification time changes (hook.py calls `refresh()` at the top of every execute_model); a key
there overrides the environment, and deleting the file (or a key) restores the environment's value.

Only code that runs eagerly (prompt chunks: more rows than the largest captured graph) may read a flag through `get`:
a decode step replays CUDA graphs captured at start-up, which keep the start-up value of every flag they depend on.
"""
import json
import os

PATH = os.environ.get("K31_FLAGS_FILE", "/kl/k31-flags.json")
_over: dict = {}
_seen = [None]                       # the control file's mtime at the last read (None: no file)
_log = [None]


def refresh() -> bool:
    """Re-read the control file if it changed. Returns True when the overrides changed."""
    try:
        m = os.stat(PATH).st_mtime_ns
    except OSError:
        m = None
    if m == _seen[0]:
        return False
    _seen[0] = m
    new = {}
    if m is not None:
        try:
            with open(PATH) as f:
                new = {str(k): str(v) for k, v in json.load(f).items() if str(k).startswith("K31_")}
        except (OSError, ValueError, AttributeError):
            return False                 # a half-written file: keep the old overrides, retry at the next change
    if new == _over:
        return False
    _over.clear()
    _over.update(new)
    if _log[0] is not None:
        _log[0](f"runtime flags from {PATH}: {new if new else 'none (environment values)'}")
    return True


def get(name: str, default: str = "") -> str:
    """The flag's current value: the control file's, else the environment's, else `default`."""
    v = _over.get(name)
    return v if v is not None else os.environ.get(name, default)


def on(name: str, default: str = "0") -> bool:
    return get(name, default) == "1"


def num(name: str, default):
    v = get(name, str(default))
    try:
        return type(default)(v) if v != "" else default
    except ValueError:
        return default
