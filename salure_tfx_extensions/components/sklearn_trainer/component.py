"""A trainer component for SKLearn models"""

from typing import Optional, Text, Dict, Any, Union

from tfx import types
from tfx.components.base import base_component
from tfx.components.base import executor_spec
from tfx.types import standard_artifacts
from salure_tfx_extensions.types import standard_artifacts as stfxe_artifacts
from tfx.types import artifact_utils
from tfx.proto import example_gen_pb2
from tfx.components.example_gen import utils
from tfx.types import channel_utils
from salure_tfx_extensions.components.sklearn_trainer import executor
from salure_tfx_extensions.types.component_specs import SKLearnTrainerSpec


class SKLearnTrainer(base_component.BaseComponent):
    """A component which trains SKLearn models using their 'fit' function"""

    SPEC_CLASS = SKLearnTrainerSpec
    EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(executor.Executor)

    def __init__(self,
                 examples: types.Channel,
                 model_pickle: Text,
                 supervised: bool = True,
                 # module_file: Text,
                 instance_name: Optional[Text] = None,
                 enable_cache: Optional[bool] = None):
        """

        :param examples: A TFX Channel of type 'Examples'
        :param module_file: A path to a module file containing the model
        """

        # TODO: Allow for transformed inputs, and transformation input graph

        output = types.Channel(
            type=stfxe_artifacts.SKLearnModel, artifacts=[stfxe_artifacts.SKLearnModel()]
        )

        spec = SKLearnTrainerSpec(
            examples=examples,
            model_pickle=model_pickle
        )

        super(SKLearnTrainer, self).__init__(
            spec=spec,
            instance_name=instance_name,
            enable_cache=enable_cache,
        )
