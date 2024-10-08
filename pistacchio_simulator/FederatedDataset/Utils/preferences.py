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
    server_test_set: str = Field(...)
    max_size: Optional[float] = None
    validation_size: Optional[float] = 0
    seed: int = Field(...)


class Preferences(BaseModel):
    dataset: str = Field(...)
    data_split_config: PartitionConfig = Field(...)
