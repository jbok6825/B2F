# Compatibility shim for legacy checkpoints that import `training.configs`.
# We reuse the B2F config values.
from training.network_b2f.configs import *  # noqa: F401,F403
