import os
import importlib.machinery, importlib.util
from types import ModuleType
import absl
import dill
import base64
import apache_beam
import tensorflow
from typing import Any, Dict, List, Text
from tfx import types
from tfx.components.base import base_executor
from tfx.types import artifact_utils
from tfx.utils import io_utils
from tfx.utils import import_utils
from tfx_bsl.tfxio import tf_example_record
import tensorflow_transform.beam as tft_beam
from salure_tfx_extensions.utils import example_parsing_utils
import apache_beam as beam
import pyarrow as pa
from sklearn.pipeline import Pipeline


EXAMPLES_KEY = 'examples'
SCHEMA_KEY = 'schema'
MODULE_FILE_KEY = 'module_file'
PREPROCESSOR_PIPELINE_NAME_KEY = 'preprocessor_pipeline_name'
PREPROCESSOR_PICKLE_KEY = 'preprocessor_pickle'
TRANSFORMED_EXAMPLES_KEY = 'transformed_examples'
TRANSFORM_PIPELINE_KEY = 'transform_pipeline'

_TELEMETRY_DESCRIPTORS = ['SKLearnTransform']

DEFAULT_PIPELINE_NAME = 'pipeline'
PIPELINE_FILE_NAME = 'pipeline.pickle'


class Executor(base_executor.BaseExecutor):
    """Executor for the SKLearnTransform Component
    Reads in Examples, and extracts a Pipeline object from a module file.
    It fits the pipeline, and writes the fit pipeline and the transformed examples to file"""

    def Do(self,
           input_dict: Dict[Text, List[types.Artifact]],
           output_dict: Dict[Text, List[types.Artifact]],
           exec_properties: Dict[Text, Any]) -> None:
        """
        Args:
          input_dict:
            - examples: Tensorflow Examples
          exec_properties:
            - preprocessor_pickle: A pickle string of the preprocessor
            - module_file: String file path to a module file
            - preprocessor_pipeline_name: The name of the pipeline object in the specified module file
          output_dict:
            - transformed_examples: The transformed Tensorflow Examples
            - transform_pipeline: A trained SKLearn Pipeline
        """

        self._log_startup(input_dict, output_dict, exec_properties)

        if not (len(input_dict[EXAMPLES_KEY]) == 1):
            raise ValueError('input_dict[{}] should only contain one artifact'.format(EXAMPLES_KEY))
        if bool(exec_properties['preprocessor_pickle']) == bool(exec_properties['module_file']):
            raise ValueError('Could not determine which preprocessor to use, both or neither of the module file and a'
                             'preprocessor pickle were provided')

        use_module_file = bool(exec_properties['module_file'])

        examples_artifact = input_dict[EXAMPLES_KEY][0]
        examples_splits = artifact_utils.decode_split_names(examples_artifact.split_names)

        train_and_eval_split = ('train' in examples_splits and 'eval' in examples_splits)
        single_split = ('single_split' in examples_artifact.uri)

        if train_and_eval_split == single_split:
            raise ValueError('Couldn\'t determine which input split to fit the pipeline on. '
                             'Exactly one split between \'train\' and \'single_split\' should be specified.')

        train_split = 'train' if train_and_eval_split else 'single_split'

        train_uri = os.path.join(examples_artifact.uri, train_split)
        absl.logging.info('train_uri: {}'.format(train_uri))

        # Load in the schema
        schema_path = io_utils.get_only_uri_in_dir(
            artifact_utils.get_single_uri(input_dict[SCHEMA_KEY]))
        schema = io_utils.SchemaReader().read(schema_path)
        absl.logging.info('schema: {}'.format(schema))

        if use_module_file:
            # Load in the specified module file
            try:
                spec = importlib.util.spec_from_file_location('user_module', exec_properties[MODULE_FILE_KEY])
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                # Load in preprocessor
                sklearn_pipeline = getattr(module, exec_properties[PREPROCESSOR_PIPELINE_NAME_KEY])
            except IOError:
                raise ImportError('{} in {} not found'.format(
                    exec_properties[PREPROCESSOR_PIPELINE_NAME_KEY], exec_properties[MODULE_FILE_KEY]))

            sklearn_pipeline = import_utils.import_func_from_source(
                exec_properties[MODULE_FILE_KEY],
                exec_properties[PREPROCESSOR_PIPELINE_NAME_KEY]
            )
        else:  # use the provided pickle
            # This way a pickle bytestring could be sent over json
            sklearn_pipeline = dill.loads(base64.decodebytes(exec_properties['preprocessor_pickle'].encode('utf-8')))

        absl.logging.info('pipeline: {}'.format(sklearn_pipeline))

        data = example_parsing_utils.from_tfrecords(train_uri, schema)
        df = example_parsing_utils.to_pandas(data)
        absl.logging.info('dataframe head: {}'.format(df.head().to_string()))

        # Fit the pipeline
        sklearn_pipeline.fit(df)

        absl.logging.info(sklearn_pipeline)
        absl.logging.info(output_dict[TRANSFORM_PIPELINE_KEY])
        with open(os.path.join(output_dict[TRANSFORM_PIPELINE_KEY], PIPELINE_FILE_NAME)) as f:
            dill.dump(sklearn_pipeline, f)

        # Scrap the beam pipeline for this component get the pickled SKLearn Pipeline to work

        # with self._make_beam_pipeline() as pipeline:
        #     with tft_beam.Context(
        #             use_deep_copy_optimization=True):
        #         absl.logging.info('Loading Training Examples')
        #         train_input_uri = io_utils.all_files_pattern(train_uri)
        #
        #         input_tfxio = tf_example_record.TFExampleRecord(
        #             file_pattern=train_input_uri,
        #             telemetry_descriptors=_TELEMETRY_DESCRIPTORS,
        #             schema=schema
        #         )
        #
        #         absl.logging.info(input_dict)
        #         absl.logging.info(output_dict)
        #         absl.logging.info('uri: {}'.format(train_uri))
        #         absl.logging.info('input_uri: {}'.format(train_input_uri))
        #
        #         training_data_recordbatch = pipeline | 'TFXIORead Train Files' >> input_tfxio.BeamSource()
        #         training_data_recordbatch | 'Logging data from Train Files' >> beam.Map(absl.logging.info)
        #
        #         training_data = (
        #             training_data_recordbatch
        #             # | 'Recordbatches to Table' >> beam.CombineGlobally(
        #             #     example_parsing_utils.RecordBatchesToTable())
        #             | 'Aggregate RecordBatches' >> beam.CombineGlobally(
        #                 beam.combiners.ToListCombineFn())
        #             # Work around non-picklability for pa.Table.from_batches
        #             | 'To Pyarrow Table' >> beam.Map(lambda x: pa.Table.from_batches(x))
        #             | 'To Pandas DataFrame' >> beam.Map(lambda x: x.to_pandas())
        #         )
        #
        #         training_data | 'Logging Pandas DataFrame' >> beam.Map(
        #             lambda x: absl.logging.info('dataframe: {}'.format(x)))
        #         training_data | 'Log DataFrame head' >> beam.Map(lambda x: print(x.head().to_string()))
        #
        #         # fit_preprocessor = training_data | 'Fit Preprocessing Pipeline' >> beam.ParDo(
        #         #     FitPreprocessingPipeline(), beam.pvalue.AsSingleton(sklearn_pipeline))
        #
        #         fit_preprocessor = training_data | 'Fit Preprocessing Pipeline' >> FitPreprocessingPipeline(
        #             sklearn_pipeline)
        #
        #         fit_preprocessor | 'Logging Fit Preprocessor' >> beam.Map(absl.logging.info)


def import_pipeline_from_source(source_path: Text, pipeline_name: Text) -> Pipeline:
    """Imports an SKLearn Pipeline object from a local source file"""

    try:
        loader = importlib.machinery.SourceFileLoader(
            fullname='user_module',
            path=source_path,
        )
        user_module = ModuleType(loader.name)
        loader.exec_module(user_module)
        return getattr(user_module, pipeline_name)
    except IOError:
        raise ImportError('{} in {} not found in import_func_from_source()'.format(
            pipeline_name, source_path))


# class SKLearnPipeline(object):
#     """This exists to appease the python pickling Gods"""
#
#     def __init__(self, pipeline):
#         self._pipeline = pipeline
#
#     def fit(self, dataframe):
#         self._pipeline.fit(dataframe)
#
#     @property
#     def pipeline(self):
#         return self._pipeline


class FitPreprocessingPipeline(beam.PTransform):
    def __init__(self, pipeline):
        self.pipeline = pipeline  # SKLearnPipeline object
        super(FitPreprocessingPipeline, self).__init__()

    # def process(self, matrix, pipeline, *args, **kwargs):
    #     pipeline.fit(matrix)
    #     return pipeline

    def expand(self, dataframe):
        """Fits an SKLearn Pipeline object

        Returns:
            A fit SKLearn Pipeline
        """

        self.pipeline.fit(dataframe)
        return [self.pipeline]


class TransformUsingPipeline(beam.DoFn):
    """Returns preprocessed inputs"""
    # TODO

    def __init__(self, pipeline: Pipeline):
        self.pipeline = pipeline
        super(TransformUsingPipeline, self).__init__()

    def process(self, element, *args, **kwargs):
        pass

