from mars.mars_pipeline import MarsPipeline, MarsPipelineConfig
from mars.models.scene_graph import SceneGraphModel, SceneGraphModelConfig

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Type, Union, cast

@dataclass
class SyntheticDataPipelineConfig(MarsPipelineConfig):
    """Configuration for the synthetic data pipeline."""

    _target: Type = field(default_factory=lambda: SyntheticDataPipeline)
    """target class to instantiate"""
    model: SceneGraphModelConfig = field(default_factory=lambda: SceneGraphModelConfig())
    """Configuration for the scene graph model."""


class SyntheticDataPipeline(MarsPipeline):
    pass

    # TODO: Implement/ Adjust pipeline

def get_background_model(checkpoint_path):
    """Extract background model from MARS checkpoint."""
    pass

def get_object_models(checkpoint_path):
    """Extract object models from MARS checkpoint."""
    pass



@dataclass 
class SyntheticSceneGraphModelConfig(SceneGraphModelConfig):
    """Configuration for the synthetic scene graph model."""

    _target: Type = field(default_factory=lambda: SyntheticSceneGraphModel)


class SyntheticSceneGraphModel(SceneGraphModel):
    """Synthetic scene graph model."""
    
    pass