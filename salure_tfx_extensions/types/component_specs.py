"""Standard Component specs for the salure_tfx_extensions library"""

from tfx.types import ComponentSpec
from tfx.types.component_spec import ChannelParameter
from tfx.types import standard_artifacts


class BaseSpec(ComponentSpec):
    """Salure_tfx_extensions BaseComponent spec"""

    PARAMETERS = {}
    INPUTS = {
        'examples': ChannelParameter(type=standard_artifacts.Examples)
    }
    OUTPUTS = {
        'examples': ChannelParameter(type=standard_artifacts.Examples)
    }

