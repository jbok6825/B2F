# Compatibility shim: re-export the updated TransformerDecoder for legacy checkpoints.
from training.network_b2f.module.TransformerDecoder import *  # noqa: F401,F403

