"""Loaded by every Python process of the vLLM container through PYTHONPATH (the image's own sitecustomize only
installs apport's exception hook, which this one keeps). K31_ENGINE=1 installs derpk31pro's MegaMoE hook."""
import os

try:
    import apport_python_hook
except ImportError:
    pass
else:
    apport_python_hook.install()

if os.environ.get("K31_ENGINE") == "1":
    from k31_engine import hook as _k31_hook
    _k31_hook.install()
