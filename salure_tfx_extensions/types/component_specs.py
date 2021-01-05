"""Standard Component specs for the salure_tfx_extensions library"""

from typing import Text

from tfx.types import ComponentSpec
from tfx.types.component_spec import ChannelParameter, ExecutionParameter
from tfx.types import standard_artifacts
from tfx.types.artifact import Artifact
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


class MySQLPusherSpec(ComponentSpec):
    """Salure_tfx_extensions MySQLPusher spec"""

    PARAMETERS = {
        'module_file': ExecutionParameter(type=(bytes, Text)),
    }

    INPUTS = {
        'inference_result': ChannelParameter(type=standard_artifacts.InferenceResult),
    }

    OUTPUTS = dict()


class SKLearnTrainerSpec(ComponentSpec):
    """Salure_tfx_extensions SKLearnTrainer spec"""

    PARAMETERS = {
        'model_pickle': ExecutionParameter(type=(bytes, Text)),
        'label_name': ExecutionParameter(type=(str, Text)),  # If None: unsupervised
    }
    INPUTS = {
        'examples': ChannelParameter(type=standard_artifacts.Examples),
        'schema': ChannelParameter(type=standard_artifacts.Schema),
    }
    OUTPUTS = {
        'transformed_examples': ChannelParameter(type=standard_artifacts.Examples),
        'model': ChannelParameter(type=stfxe_artifacts.SKLearnModel),
    }


class SKLearnTransformSpec(ComponentSpec):
    """Salure_tfx_extensions SKLearnTransform spec"""

    PARAMETERS = {
        # 'module_file': ExecutionParameter(type=(str, Text), optional=True),
        # 'preprocessor_pipeline_name': ExecutionParameter(type=(str, Text), optional=True),
        'preprocessor_pickle': ExecutionParameter(type=(str, Text))
        # 'data_format': ExecutionParameter(type=(str, Text), optional=True),  # Default will be pandas
    }
    INPUTS = {
        'examples': ChannelParameter(type=standard_artifacts.Examples),
        'schema': ChannelParameter(type=standard_artifacts.Schema)
    }
    OUTPUTS = {
        'transformed_examples': ChannelParameter(type=standard_artifacts.Examples),
        'transform_pipeline': ChannelParameter(type=stfxe_artifacts.SKLearnPrepocessor)
    }


class PusherSpec(ComponentSpec):
    """Salure_tfx_extensions Pusher spec"""

    PARAMETERS = {
        'push_destination': ExecutionParameter(type=(str, Text)),
        'timestamp_versioning': ExecutionParameter(type=bool)
    }
    INPUTS = {
        'pushable': ChannelParameter(type=Artifact),  # Allow for any artifact type to be pushed
        'model_blessing': ChannelParameter(type=standard_artifacts.ModelBlessing, optional=True),
        'infra_blessing': ChannelParameter(type=standard_artifacts.InfraBlessing, optional=True)
    }
    OUTPUTS = {
        'pushed_model': ChannelParameter(type=standard_artifacts.PushedModel)
    }
