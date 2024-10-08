from typing import Optional

from pydantic import BaseModel, Field  # , field_validator
from pydantic.dataclasses import dataclass


class PartitionConfig(BaseModel):
    split_type_clusters: Optional[str] = None
    split_type_nodes: Optional[str] = None
    num_classes: int = Field(...)
    num_nodes: Optional[int] = None
    num_clusters: Optional[int] = None
    alpha: Optional[int] = None
    percentage_configuration: Optional[dict] = None
    store_path: str = Field(...)
    server_test_set: str = Field(...)
    max_size: Optional[float] = None
    validation_size: Optional[float] = 0
    seed: int = Field(...)


class P2PConfig(BaseModel):
    fl_rounds: int
    local_training_epochs: int
    round_mixed_mode: Optional[int] = 0
    differential_privacy: bool
    differential_privacy_mixed_mode: Optional[bool] = False
    epsilon: Optional[float] = None
    noise_multiplier: Optional[float] = None
    epsilon_mixed: Optional[float] = None
    lr: float = None
    batch_size: int = None


class ServerConfig(BaseModel):
    fl_rounds: int = Field(...)
    local_training_epochs: int
    differential_privacy: bool
    mixed_mode: bool
    total_mixed_rounds: Optional[int] = 0
    epsilon: Optional[float] = None
    noise_multiplier: Optional[float] = None
    lr: float = None
    batch_size: int = None


class HyperparametersConfig(BaseModel):
    max_phisical_batch_size: int
    delta: Optional[float] = None
    max_grad_norm: float = Field(...)
    optimizer: str = Field(...)


class WandbConfig(BaseModel):
    tags: list[str]
    name: str
    project_name: str
    sweep: bool


@dataclass
class Preferences:
    dataset: str = Field(...)
    task_type: str = Field(...)
    debug: bool = Field(...)
    wandb: bool = Field(...)
    public_private_experiment: bool = Field(...)
    pool_size: int = Field(...)
    data_split_config: PartitionConfig = Field(...)
    p2p_config: Optional[P2PConfig] = None
    server_config: Optional[ServerConfig] = None
    hyperparameters_config: Optional[HyperparametersConfig] = Field(...)
    gpu_config: list[str] = Field(...)
    wandb_config: Optional[WandbConfig] = None
    dataset_p2p: Optional[str] = None
    dataset_server: Optional[str] = None
