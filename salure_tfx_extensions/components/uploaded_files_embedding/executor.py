"""Custom LC Embedding Component Executor"""

import os
import apache_beam as beam
import tensorflow as tf
from tfx.components.base import base_executor
from tfx.types import artifact_utils
from tfx.utils import io_utils
import pandas as pd


feature_description = {
	"bedrag":tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
	"boekjaar":tf.io.FixedLenFeature([], tf.int64, default_value=0),
	"cao_code":tf.io.FixedLenFeature([], tf.string, default_value=''),
	"dagen_per_week":tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
	"expired_rooster":tf.io.FixedLenFeature([], tf.int64, default_value=0),
	"fte":tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
	"fte_di":tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
	"fte_do":tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
	"fte_ma":tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
	"fte_vr":tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
	"fte_wo":tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
	"full_time_contract":tf.io.FixedLenFeature([], tf.int64, default_value=0),
# 	"hoofddienstverband":tf.io.SparseFeature([], tf.string, default_value=''),
	"looncomponent_extern_nummer":tf.io.FixedLenFeature([], tf.int64, default_value=0),
	"medewerker_id":tf.io.FixedLenFeature([], tf.int64, default_value=0),
	"new_rooster":tf.io.FixedLenFeature([], tf.int64, default_value=0),
	"part_time_contract":tf.io.FixedLenFeature([], tf.int64, default_value=0),
	"periode":tf.io.FixedLenFeature([], tf.int64, default_value=0),
	"temp_contract":tf.io.FixedLenFeature([], tf.int64, default_value=0),
	"trainee_time_contract":tf.io.FixedLenFeature([], tf.int64, default_value=0),
	"type_contract":tf.io.FixedLenFeature([], tf.int64, default_value=0),
	"type_medewerker":tf.io.FixedLenFeature([], tf.string, default_value=''),
	"uren_per_week":tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
	"werkgever_id":tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'hours_days_km':tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'costs':tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'is_fixed':tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'is_variable':tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'declaration':tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'allowance':tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'travel_related':tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'company_car':tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'overtime':tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'leaves':tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'health':tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'insurance':tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'pension':tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'overig':tf.io.FixedLenFeature([], tf.int64, default_value=0)
}


def uploadedfilesmapping(input_data, mapping_uri):
    df = pd.read_csv(os.path.join(mapping_uri, 'uploaded_files.csv'))
    df['medewerker_id'] = df['medewerker_id'].apply(lambda x: str(x)[0:-2] if str(x)[-2:] == str(".0") else str(x))
    df['looncomponent_extern_nummer'] = df['looncomponent_extern_nummer'].apply(
        lambda x: str(x)[0:-2] if str(x)[-2:] == str(".0") else str(x))
    data = tf.io.parse_single_example(input_data, feature_description)

    medewerker_id = data['medewerker_id'].numpy()
    looncomponent_extern_nummer = data['looncomponent_extern_nummer'].numpy()
    adjusted_amount = df[(df['medewerker_id'] == str(medewerker_id)) & \
                         (df['looncomponent_extern_nummer'] == str(looncomponent_extern_nummer)) & \
                         (df['boekjaar'] == data['boekjaar'].numpy()) & \
                         (df['periode'] == data['periode'].numpy())
                         ]['adjusted_amount']

    if adjusted_amount.shape[0] > 0:
        data['adjusted_amount'] = tf.constant(adjusted_amount.values[0])
    else:
        data['adjusted_amount'] = tf.constant(0)

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

        with beam.Pipeline() as pipeline:
            train_data = (
                    pipeline
                    | 'ReadData' >> beam.io.ReadFromTFRecord(
                file_pattern=io_utils.all_files_pattern(input_examples_uri))
                    | 'Mapping Wage Components' >> beam.Map(uploadedfilesmapping, mapping_uri)
                    | 'SerializeExample' >> beam.Map(lambda x: x.SerializeToString())
                    | 'WriteAugmentedData' >> beam.io.WriteToTFRecord(
                os.path.join(train_output_examples_uri, "uploadedfiles_embedded_data"), file_name_suffix='.gz'))

        with beam.Pipeline() as pipeline:
            eval_data = (
                    pipeline
                    | 'ReadData' >> beam.io.ReadFromTFRecord(
                file_pattern=io_utils.all_files_pattern(eval_input_examples_uri))
                    | 'Mapping Wage Components' >> beam.Map(uploadedfilesmapping, mapping_uri)
                    | 'SerializeExample' >> beam.Map(lambda x: x.SerializeToString())
                    | 'WriteAugmentedData' >> beam.io.WriteToTFRecord(
                os.path.join(eval_output_examples_uri, "uploadedfiles_embedded_data"), file_name_suffix='.gz'))