from typing import Optional
from distributed.engine import DistributedEngine
from inference.engine import InferenceEngine


class BaseTrainer:
    r""" """

    # `distributed_engine` manages the distributed training framework backends,
    # TODO: for current version, only FSDPv2 is supported
    distributed_engine: Optional[DistributedEngine] = None

    # `inference_engine` calls the inferencer backends for rollout in RL training.
    inference_engine: Optional[InferenceEngine] = None
