# TODO: fill in, to which base type it gets converted
"""Base executor, reads Examples, converts them to <fill in>, **writes them to logs** and back to Example"""

import os
import absl
import apache_beam as beam
import tensorflow as tf
from typing import Any, Dict, List, Text
from tfx import types
from tfx.components.base import base_executor
from tfx.types import artifact_utils
from tfx.utils import io_utils
from tfx.utils import path_utils
from tfx_bsl.tfxio import tf_example_record

EXAMPLES_KEY = 'examples'
_TELEMETRY_DESCRIPTORS = ['BaseComponent']
DEFAULT_FILE_NAME = 'data_tfrecord'


class Executor(base_executor.BaseExecutor):
    """Executor for the BaseComponent boilerplate.
    Will read in Examples, convert them rows, and then back writing them to file as examples"""

    def Do(self, input_dict: Dict[Text, List[types.Artifact]],
           output_dict: Dict[Text, List[types.Artifact]],
           exec_properties: Dict[Text, Any]) -> None:
        """Validate current model against last blessed model.
    Args:
      input_dict: Input dict from input key to a list of Artifacts.
        - examples: Tensorflow Examples
      output_dict: Output dict from output key to a list of Artifacts.
        - examples: Tensorflow Examples
      exec_properties: A dict of execution properties.
        In this case there are no items in exec_properties, as stated by BaseComponentSpec
    Returns:
      None
    """
        self._log_startup(input_dict, output_dict, exec_properties)

        if EXAMPLES_KEY not in input_dict:
            raise ValueError('\'{}\' is missing from input_dict'.format(EXAMPLES_KEY))

        split_uris = []
        for artifact in input_dict[EXAMPLES_KEY]:
            for split in artifact_utils.decode_split_names(artifact.split_names):
                uri = os.path.join(artifact.uri, split)
                split_uris.append((split, uri))

        with self._make_beam_pipeline() as pipeline:
            for split, uri in split_uris:
                absl.logging.info('Loading examples for split {}'.format(split))
                input_uri = io_utils.all_files_pattern(uri)
                input_tfxio = tf_example_record.TFExampleRecord(
                    file_pattern=input_uri,
                    telemetry_descriptors=_TELEMETRY_DESCRIPTORS
                )
                output_path = artifact_utils.get_split_uri(output_dict[EXAMPLES_KEY],
                                                          split)

                # loading the data and displaying
                data = pipeline | 'TFXIORead[{}]'.format(split) >> input_tfxio.BeamSource()
                data | 'Printing data from {}'.format(split) >> beam.Map(absl.logging.info)

                data | 'WriteSplit[{}]'.format(split) >> _WriteSplit(output_path)


@beam.ptransform_fn
@beam.typehints.with_input_types(bytes)
@beam.typehints.with_output_types(beam.pvalue.PDone)
def _WriteSplit(example_split: beam.pvalue.PCollection,
                output_split_path: Text) -> beam.pvalue.PDone:
    """Shuffles and writes output split."""
    return (example_split
            | 'Shuffle' >> beam.transforms.Reshuffle()
            | 'Write' >> beam.io.WriteToTFRecord(
                os.path.join(output_split_path, DEFAULT_FILE_NAME),
                file_name_suffix='.gz'))

