# Compatibility shim for legacy checkpoints saved with `training.network.*`
# Re-exports the updated B2F network implementation.
from training.network_b2f.Network import Network  # noqa: F401

