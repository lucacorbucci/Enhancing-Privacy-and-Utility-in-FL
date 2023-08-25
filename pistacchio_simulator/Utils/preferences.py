from typing import Optional

from pydantic import BaseModel, Field  # , field_validator


class PartitionConfig(BaseModel):
    split_type_clusters: Optional[str] = None
    split_type_nodes: Optional[str] = None
    num_classes: int = Field(...)
    num_nodes: Optional[int] = None
    num_clusters: Optional[int] = None
    alpha: Optional[int] = None
    percentage_configuration: Optional[dict] = None
    store_path: str = Field(...)

class P2PConfig(BaseModel):
    fl_rounds: int
    local_training_epochs: int
    round_mixed_mode: Optional[int] = 0
    differential_privacy: bool 
    differential_privacy_mixed_mode: Optional[bool] = False
    epsilon: Optional[float] = None
    noise_multiplier: Optional[float] = None 
    delta: Optional[float] = None
    epsilon_mixed: Optional[float] = None
class ServerConfig(BaseModel):
    fl_rounds = int
    local_training_epochs = int
    differential_privacy: bool
    mixed_model: bool 
    FL_rounds: int 
    total_mixed_rounds: Optional[int] = 0
    epsilon: Optional[float] = None
    noise_multiplier: Optional[float] = None 
    delta: Optional[float] = None
class HyperparametersConfig(BaseModel):
    batch_size: int
    lr: float
    max_phisical_batch_size: int 
class WandbConfig(BaseModel):
    tags: list[str]
    name: str 
    project_name: str

class Preferences(BaseModel):
    dataset: str = Field(...)
    public_private_experiment: bool = Field(...)
    pool_size: int = Field(...)
    data_split_config: PartitionConfig = Field(...)
    p2p_config: Optional[P2PConfig] = None
    server_config: Optional[ServerConfig] = None
    hyperparameters_config: Optional[HyperparametersConfig] = Field(...)
    gpu_config: list[str] = Field(...)
    wandb_config = Optional[WandbConfig] = Field(...)
