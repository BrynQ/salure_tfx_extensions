"""Standard Component specs for the salure_tfx_extensions library"""

from tfx.types import ComponentSpec
from tfx.types.component_spec import ChannelParameter, ExecutionParameter
from tfx.types import standard_artifacts
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

