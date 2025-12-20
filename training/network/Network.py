# Compatibility shim for legacy checkpoints that reference `training.network.Network`.
# The actual implementation now lives in `training.network_b2f.Network`.
from training.network_b2f.Network import Network  # noqa: F401

