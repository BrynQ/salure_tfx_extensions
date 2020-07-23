"""Helper functions for handling sklearn models"""

from typing import Text, Optional, List
import apache_beam as beam
from tensorflow_metadata.proto.v0 import schema_pb2
from tfx_bsl.tfxio import tf_example_record
from salure_tfx_extensions.utils import example_parsing_utils
import pyarrow as pa
import dill


class WriteSKLearnModelToFile(beam.PTransform):
    def __init__(self, file_path):
        self.file_path = file_path
        super(WriteSKLearnModelToFile, self).__init__()

    def expand(self, model):
        return (
            model
            | 'Pickle Model' >> beam.FlatMap(dill.dumps)
            | 'Write Pickled Model To File' >> beam.io.WriteToText(
                self.file_path,
                num_shards=1,
                shard_name_template=''))


class ReadTFRecordToPandas(beam.PTransform):

    def __init__(self,
                 file_pattern: Text,
                 schema: schema_pb2.Schema,
                 split_name: Text,
                 telemetry_descriptors: Optional[List[Text]] = None):
        """

        :param file_pattern: A file pattern like 'a/b/c/*' to get everything in the c folder
        :param schema:
        :param split_name: The name of the data split to name the read operations with
        :param telemetry_descriptors:
        """

        self.tfxio = tf_example_record.TFExampleRecord(
            file_pattern=file_pattern,
            telemetry_descriptors=telemetry_descriptors,
            schema=schema
        )
        self.split_name = split_name
        super(ReadTFRecordToPandas, self).__init__()

    def expand(self, pipeline):
        return (
            pipeline
            | 'TFXIORead {} Files'.format(self.split_name) >> self.tfxio.BeamSource()
            | 'Aggregate {} Recordbatches'.format(self.split_name) >> beam.CombineGlobally(
                beam.combiners.ToListCombineFn())
            # lambda so it's pickleable
            | '{} Data to Pyarrow Table'.format(self.split_name) >> beam.Map(lambda x: pa.Table.from_batches(x))
            | '{} Data to Pandas DataFrame'.format(self.split_name) >> beam.Map(lambda x: x.to_pandas()))


class WriteDataFrame(beam.PTransform):

    def __init__(self, file_path):
        self.file_path = file_path
        super(WriteDataFrame, self).__init__()

    def expand(self, df):
        return (
            df
            | 'DataFrame to dicts' >> beam.FlatMap(lambda x: x.to_dict('records'))
            | 'Dicts to Examples' >> beam.Map(example_parsing_utils.dict_to_example)
            | 'Write Examples' >> example_parsing_utils.WriteSplit(self.file_path))
