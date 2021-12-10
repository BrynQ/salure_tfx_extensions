"""Custom Uploadedfiles Embedding Component Executor"""

import os
import apache_beam as beam
import tensorflow as tf
from tfx.components.base import base_executor
from tfx.types import artifact_utils
from tfx.utils import io_utils, json_utils
import pandas as pd
import json


def uploadedfilesmapping(input_data, mapping_uri, feature_description, df):
    data = tf.io.parse_single_example(input_data, feature_description)

    medewerker_id = data['medewerker_id'].numpy().decode("utf-8")
    looncomponent_extern_nummer = data['looncomponent_extern_nummer'].numpy().decode("utf-8")
    adjusted_amount = df[(df['medewerker_id'] == str(medewerker_id)) & \
                         (df['looncomponent_extern_nummer'] == str(looncomponent_extern_nummer)) & \
                         (df['boekjaar'] == data['boekjaar'].numpy()) & \
                         (df['periode_uitgevoerd'] == data['periode_uitgevoerd'].numpy())
                         ]['adjusted_amount']

    if adjusted_amount.shape[0] > 0:
        data['adjusted_amount'] = tf.constant(adjusted_amount.values[0], dtype=tf.float32)
    else:
        data['adjusted_amount'] = tf.constant(0, dtype=tf.float32)

    feature = {}
    for key, value in data.items():
        if value.dtype in ['float64', 'float32']:
            feature[key] = tf.train.Feature(float_list=tf.train.FloatList(value=[value.numpy()]))
        elif value.dtype in ['int64', 'int32']:
            feature[key] = tf.train.Feature(int64_list=tf.train.Int64List(value=[value.numpy()]))
        else:
            feature[key] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.numpy()]))
    return tf.train.Example(features=tf.train.Features(feature=feature))


class Executor(base_executor.BaseExecutor):

    def Do(self, input_dict, output_dict, exec_properties):
        self._log_startup(input_dict, output_dict, exec_properties)
        mapping_uri = input_dict['mapping_data'][0].uri
        split_names = artifact_utils.decode_split_names(input_dict['input_data'][0].split_names)
        output_dict['output_data'][0].split_names = (artifact_utils.encode_split_names(split_names))

        input_examples_uri = artifact_utils.get_split_uri(input_dict['input_data'], 'train')
        eval_input_examples_uri = artifact_utils.get_split_uri(input_dict['input_data'], 'eval')

        train_output_examples_uri = os.path.join(artifact_utils.get_single_uri(output_dict['output_data']), 'train')
        eval_output_examples_uri = os.path.join(artifact_utils.get_single_uri(output_dict['output_data']), 'eval')

        with open(json_utils.loads(exec_properties['feature_description']), "rb") as read_file:
            feature_description = json.load(read_file)

        keys = [k for k in feature_description.keys()]
        values = [eval(v) for v in feature_description.values()]
        feature_description = dict(zip(keys, values))

        # Set up the Dataframe
        df = pd.read_csv(os.path.join(mapping_uri, 'uploaded_files.csv'))
        df['medewerker_id'] = df['medewerker_id'].apply(lambda x: str(x)[0:-2] if str(x)[-2:] == str(".0") else str(x))
        df['looncomponent_extern_nummer'] = df['looncomponent_extern_nummer'].apply(
            lambda x: str(x)[0:-2] if str(x)[-2:] == str(".0") else str(x))

        with beam.Pipeline() as pipeline:
            train_data = (
                    pipeline
                    | 'ReadData' >> beam.io.ReadFromTFRecord(
                        file_pattern=io_utils.all_files_pattern(input_examples_uri))
                    | 'Mapping Wage Components' >> beam.Map(
                        uploadedfilesmapping,
                        mapping_uri,
                        feature_description=feature_description,
                        df=df)
                    | 'SerializeExample' >> beam.Map(lambda x: x.SerializeToString())
                    | 'WriteAugmentedData' >> beam.io.WriteToTFRecord(
                        os.path.join(train_output_examples_uri, "uploadedfiles_embedded_data"), file_name_suffix='.gz'))

        with beam.Pipeline() as pipeline:
            eval_data = (
                    pipeline
                    | 'ReadData' >> beam.io.ReadFromTFRecord(
                        file_pattern=io_utils.all_files_pattern(eval_input_examples_uri))
                    | 'Mapping Wage Components' >> beam.Map(
                        uploadedfilesmapping,
                        mapping_uri,
                        feature_description=feature_description,
                        df=df)
                    | 'SerializeExample' >> beam.Map(lambda x: x.SerializeToString())
                    | 'WriteAugmentedData' >> beam.io.WriteToTFRecord(
                        os.path.join(eval_output_examples_uri, "uploadedfiles_embedded_data"), file_name_suffix='.gz'))