"""Custom TFX Pusher component definition."""
from typing import Any, Dict, Optional, Text, Union

import absl

from tfx import types
from tfx.components.base import base_component
from tfx.components.base import executor_spec
from salure_tfx_extensions.components.pusher import executor
from salure_tfx_extensions.types.component_specs import PusherSpec
from tfx.proto import pusher_pb2
# from tfx.types import standard_artifacts
# from tfx.types.standard_component_specs import PusherSpec
# from tfx.utils import json_utils


# TODO(b/133845381): Investigate other ways to keep push destination converged.
class Pusher(base_component.BaseComponent):
    """A TFX component to push validated TensorFlow models to a model serving platform.
  The `Pusher` component can be used to push an validated SavedModel from output
  of the [Trainer component](https://www.tensorflow.org/tfx/guide/trainer) to
  [TensorFlow Serving](https://www.tensorflow.org/tfx/serving).  The Pusher
  will check the validation results from the [ModelValidator
  component](https://www.tensorflow.org/tfx/guide/model_validator)
  before deploying the model.  If the model has not been blessed, then the model
  will not be pushed.
  *Note:* The executor for this component can be overriden to enable the model
  to be pushed to other serving platforms than tf.serving.  The [Cloud AI
  Platform custom
  executor](https://github.com/tensorflow/tfx/tree/master/tfx/extensions/google_cloud_ai_platform/pusher)
  provides an example how to implement this.
  ## Example
  ```
    # Checks whether the model passed the validation steps and pushes the model
    # to a file destination if check passed.
    pusher = Pusher(
        model=trainer.outputs['model'],
        model_blessing=model_validator.outputs['blessing'],
        push_destination=pusher_pb2.PushDestination(
            filesystem=pusher_pb2.PushDestination.Filesystem(
                base_directory=serving_model_dir)))
  ```
  """

    SPEC_CLASS = PusherSpec
    EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(executor.Executor)

    def __init__(
            self,
            pushable: types.Channel = None,
            model_blessing: types.Channel = None,
            infra_blessing: Optional[types.Channel] = None,
            push_destination: Optional[Union[pusher_pb2.PushDestination,
                                             Dict[Text, Any]]] = None,
            timestamp_versioning: bool = False,
            instance_name: Optional[Text] = None):
        """Construct a Pusher component.
    Args:
      pushable: A Channel of any Artifact type.
        The folder in which the Artifact is stored, will be copied to the push_destination.
      model_blessing: Optional. A Channel of type `standard_artifacts.ModelBlessing`,
        usually produced by a ModelValidator component. _required_
      infra_blessing: Optional. A Channel of type
        `standard_artifacts.InfraBlessing`, usually produced from an
        InfraValidator component.
      push_destination: A pusher_pb2.PushDestination instance, providing info
        for tensorflow serving to load models. Optional if executor_class
        doesn't require push_destination. If any field is provided as a
        RuntimeParameter, push_destination should be constructed as a dict with
        the same field names as PushDestination proto message.
      instance_name: Optional unique instance name. Necessary if multiple Pusher
        components are declared in the same pipeline.
    """

        spec = PusherSpec(
            push_destination=push_destination,
            timestamp_versioning=timestamp_versioning,
            pushable=pushable,
            model_blessing=model_blessing,
            infra_blessing=infra_blessing)
        super(Pusher, self).__init__(
            spec=spec,
            instance_name=instance_name)
