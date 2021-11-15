"""Custom TFX Component specs for the salure_tfx_extensions library"""

from typing import Text
from tfx.types import ComponentSpec
from tfx.types.component_spec import ChannelParameter, ExecutionParameter
from tfx.types import standard_artifacts
from tfx.types.artifact import Artifact
from tfx.proto import example_gen_pb2
from salure_tfx_extensions.types import standard_artifacts as stfxe_artifacts
from salure_tfx_extensions.proto import mysql_config_pb2


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

class LCEmbeddingSpec(ComponentSpec):
    """ComponentSpec for Custom TFX Hello World Component."""

    PARAMETERS = {
      # These are parameters that will be passed in the call to create an instance of this component.
      #   'name': ExecutionParameter(type=Text),
        'mapping_file_path':ExecutionParameter(type=(str, Text)),
        'feature_description': ExecutionParameter(type=(str, Text))  # new parameter for reading f_desc_emb.json
    }
    INPUTS = {
        # This will be a dictionary with input artifacts, including URIs
        'input_data': ChannelParameter(type=standard_artifacts.Examples),
    }
    OUTPUTS = {
        # This will be a dictionary which this component will populate
        'output_data': ChannelParameter(type=standard_artifacts.Examples),
    }

class CopyFileSpec(ComponentSpec):
    """ComponentSpec for Custom TFX Hello World Component."""

    PARAMETERS = {
      # These are parameters that will be passed in the call to create an instance of this component.
#         'name': ExecutionParameter(type=Text),
        'input_path':ExecutionParameter(type=(str, Text)),
        'output_path':ExecutionParameter(type=(str, Text))
    }
    INPUTS = {
        # This will be a dictionary with input artifacts, including URIs
    }
    OUTPUTS = {
        # This will be a dictionary which this component will populate
    }

class PercentileComponentSpec(ComponentSpec):
    """ComponentSpec for Custom TFX Hello World Component."""

    PARAMETERS = {
      # These are parameters that will be passed in the call to create an instance of this component.
        'num_quantiles': ExecutionParameter(type=Text),
    }
    INPUTS = {
        'inference_result': ChannelParameter(type=standard_artifacts.InferenceResult),
    }
    OUTPUTS = {
        # This will be a dictionary which this component will populate
        'percentile_values':  ChannelParameter(type=standard_artifacts.Examples),
    }

class CsvToExampleSpec(ComponentSpec):
    """Transform component spec."""

    PARAMETERS = {
      'feature_description': ExecutionParameter(type=(str, Text)),
        'input_path': ExecutionParameter(type=(str, Text))
    }
    INPUTS = {
    }
    OUTPUTS = {
      'examples':
          ChannelParameter(type=standard_artifacts.Examples)
    }

class UploadedfilesEmbeddingSpec(ComponentSpec):
    """ComponentSpec for Custom TFX UploadedfilesEmbedding Component."""

    PARAMETERS = {
      # These are parameters that will be passed in the call to create an instance of this component.
        'feature_description': ExecutionParameter(type=(str, Text)),  # new parameter for reading f_desc_upl.json
    }
    INPUTS = {
        # This will be a dictionary with input artifacts, including URIs
        'input_data': ChannelParameter(type=standard_artifacts.Examples),
        'mapping_data': ChannelParameter(type=standard_artifacts.Examples)
    }
    OUTPUTS = {
        # This will be a dictionary which this component will populate
        'output_data': ChannelParameter(type=standard_artifacts.Examples),
    }

class MySQLPusherSpec(ComponentSpec):
    """Salure_tfx_extensions MySQLPusher spec"""

    PARAMETERS = {
        'connection_config': ExecutionParameter(type=mysql_config_pb2.MySQLConnConfig),
        'table_name': ExecutionParameter(type=(str, Text)),
        'inference_id': ExecutionParameter(type=(str, Text))
    }

    INPUTS = {
        'inference_result': ChannelParameter(type=standard_artifacts.InferenceResult),
        'percentile_values': ChannelParameter(type=standard_artifacts.Examples)
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
