# Lint as: python2, python3
# Copyright 2019 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""TFX pusher executor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from datetime import datetime
from typing import Any, Dict, List, Text

from absl import logging
import tensorflow as tf

from google.protobuf import json_format
from tfx import types
from tfx.components.base import base_executor
from tfx.components.util import model_utils
from tfx.proto import pusher_pb2
from tfx.types import artifact_utils
from tfx.utils import io_utils
from tfx.utils import path_utils

# Key for model in executor input_dict.
# MODEL_KEY = 'model'

# Key for the artifact to push in executor input_dict.
# This thing is supposed to push any artifact.
PUSHABLE_KEY = 'pushable'
# Key for model blessing in executor input_dict.
MODEL_BLESSING_KEY = 'model_blessing'
# Key for infra blessing in executor input_dict.
INFRA_BLESSING_KEY = 'infra_blessing'
# Key for pushed model in executor output_dict.
# PUSHED_MODEL_KEY = 'pushed_model'
# Key for the push destination
PUSH_DESTINATION_KEY = 'push_destination'
# Key for whether to append the push_destination with a folder with timestamp
TIMESTAMP_VERSIONING_KEY = 'timestamp_versioning'


class Executor(base_executor.BaseExecutor):

    def CheckBlessing(self, input_dict: Dict[Text, List[types.Artifact]]) -> bool:
        """Check that model is blessed by upstream validators.
    Args:
      input_dict: Input dict from input key to a list of artifacts:
        - pushable: Any artifact type of which the folder path will be copied to the push_destination
        - model_blessing: A `ModelBlessing` artifact from model validator or
          evaluator.
          Pusher looks for a custom property `blessed` in the artifact to check
          it is safe to push.
        - infra_blessing: An `InfraBlessing` artifact from infra validator.
          Pusher looks for a custom proeprty `blessed` in the artifact to
          determine whether the model is mechanically servable from the model
          server to which Pusher is going to push.
    Returns:
      True if the model is blessed by validator.
    """
        if MODEL_BLESSING_KEY in input_dict:
            model_blessing = artifact_utils.get_single_instance(
                input_dict[MODEL_BLESSING_KEY])
            if not model_utils.is_model_blessed(model_blessing):
                logging.info('Model on %s was not blessed by model validation',
                             model_blessing.uri)
                return False
        if INFRA_BLESSING_KEY in input_dict:
            infra_blessing = artifact_utils.get_single_instance(
                input_dict[INFRA_BLESSING_KEY])
            if not model_utils.is_infra_validated(infra_blessing):
                logging.info('Model on %s was not blessed by infra validator',
                             infra_blessing.uri)
                return False
        return True

    def Do(self, input_dict: Dict[Text, List[types.Artifact]],
           output_dict: Dict[Text, List[types.Artifact]],
           exec_properties: Dict[Text, Any]) -> None:
        """Push model to target directory if blessed.
    Args:
      input_dict: Input dict from input key to a list of artifacts, including:
        - pushable: Any artifact type of which the folder path will be copied to the push_destination
        - model_blessing: A `ModelBlessing` artifact from model validator or
          evaluator.
          Pusher looks for a custom property `blessed` in the artifact to check
          it is safe to push.
        - infra_blessing: An `InfraBlessing` artifact from infra validator.
          Pusher looks for a custom proeprty `blessed` in the artifact to
          determine whether the model is mechanically servable from the model
          server to which Pusher is going to push.
      output_dict: Output dict from key to a list of artifacts, including:
        - pushed_model: A list of 'ModelPushPath' artifact of size one. It will
          include the model in this push execution if the model was pushed.
      exec_properties: A dict of execution properties, including:
        - push_destination: JSON string of pusher_pb2.PushDestination instance,
          providing instruction of destination to push model.
    Returns:
      None
    """
        self._log_startup(input_dict, output_dict, exec_properties)
        # model_push = artifact_utils.get_single_instance(
        #     output_dict[PUSHED_MODEL_KEY])
        if not self.CheckBlessing(input_dict):
            # model_push.set_int_custom_property('pushed', 0)
            return
        # model_push_uri = model_push.uri
        pushable_export = artifact_utils.get_single_instance(input_dict[PUSHABLE_KEY])
        pushable_export_uri = pushable_export.uri
        logging.info('Model pushing.')
        # Copy the model to pushing uri.
        model_path = path_utils.serving_model_path(pushable_export_uri)
        # model_version = path_utils.get_serving_model_version(pushable_export_uri)
        # logging.info('Model version is %s', model_version)
        # io_utils.copy_dir(model_path, os.path.join(model_push_uri))
        # logging.info('Model written to %s.', model_push_uri)

        # Copied to a fixed outside path, which can be listened by model server.
        #
        # If model is already successfully copied to outside before, stop copying.
        # This is because model validator might blessed same model twice (check
        # mv driver) with different blessing output, we still want Pusher to
        # handle the mv output again to keep metadata tracking, but no need to
        # copy to outside path again..
        push_destination = pusher_pb2.PushDestination()
        json_format.Parse(exec_properties['push_destination'], push_destination)

        destination_kind = push_destination.WhichOneof('destination')

        if destination_kind == 'filesystem':
            serving_path = push_destination.filesystem.base_directory

            if exec_properties[TIMESTAMP_VERSIONING_KEY]:
                ts = str(int(datetime.timestamp(datetime.now())))
                serving_path = os.path.join(push_destination.filesystem.base_directory, ts)

            if tf.io.gfile.exists(serving_path):
                logging.info(
                    'Destination directory %s already exists, skipping current push.',
                    serving_path)
            else:
                # tf.serving won't load partial model, it will retry until fully copied.
                # io_utils.copy_dir(model_path, serving_path)
                io_utils.copy_dir(push_destination, serving_path)
                logging.info('Model written to serving path %s.', serving_path)

            # model_push.set_int_custom_property('pushed', 1)
            # model_push.set_string_custom_property('pushed_model', model_export_uri)
            # model_push.set_int_custom_property('pushed_model_id', model_export.id)
            logging.info('Model pushed to %s.', serving_path)

        else:
            raise NotImplementedError('Invalid push destination {}'.format(destination_kind))
