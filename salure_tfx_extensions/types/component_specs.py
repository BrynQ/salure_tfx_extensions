"""Standard Component specs for the salure_tfx_extensions library"""

from typing import Text

from tfx.types import ComponentSpec
from tfx.types.component_spec import ChannelParameter, ExecutionParameter
from tfx.types import standard_artifacts
from salure_tfx_extensions.types import standard_artifacts as stfxe_artifacts
from tfx.proto import example_gen_pb2


class BaseSpec(ComponentSpec):
    """Salure_tfx_extensions BaseComponent spec"""

    PARAMETERS = {
        'input_config': ExecutionParameter(type=example_gen_pb2.Input),
        'output_config': ExecutionParameter(type=example_gen_pb2.Output),
    }
    INPUTS = {
        'examples': ChannelParameter(type=standard_artifacts.Examples)
    }
    OUTPUTS = {
        'output_examples': ChannelParameter(type=standard_artifacts.Examples)
    }


class SKLearnTrainerSpec(ComponentSpec):
    """Salure_tfx_extensions SKLearnTrainer spec"""

    PARAMETERS = {
        'model_pickle': ExecutionParameter(type=(bytes, Text)),
        'supervised': ExecutionParameter(type=bool),
    }
    INPUTS = {
        'examples': ChannelParameter(type=standard_artifacts.Examples),
    }
    OUTPUTS = {
        'model': ChannelParameter(type=stfxe_artifacts.SKLearnModel),
    }


class SKLearnTransformSpec(ComponentSpec):
    """Salure_tfx_extensions SKLearnTransform spec"""

    PARAMETERS = {
        'module_file': ExecutionParameter(type=(str, Text)),
        'pipeline_name': ExecutionParameter(type=(str, Text), optional=True)
    }
    INPUTS = {
        'examples': ChannelParameter(type=standard_artifacts.Examples)
    }
    OUTPUTS = {
        'transformed_examples': ChannelParameter(type=standard_artifacts.Examples),
        'transform_pipeline': ChannelParameter(type=stfxe_artifacts.SKLearnPrepocessor)
    }
