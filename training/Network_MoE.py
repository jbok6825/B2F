# Compatibility wrapper for previously saved FLAME→ARKit models.
# Historic checkpoints were saved with the import path `training.Network_MoE`.
# We now host the implementation under `training.network_flame_to_arkit.Network_MoE`.
from training.network_flame_to_arkit.Network_MoE import Network  # noqa: F401
