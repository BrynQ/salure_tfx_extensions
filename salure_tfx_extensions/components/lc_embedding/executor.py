"""Custom LC Embedding Component Executor"""

import os
import apache_beam as beam
import tensorflow as tf
from tfx.components.base import base_executor
from tfx.types import artifact_utils
from tfx.utils import io_utils, json_utils
import json

def wcmapping(input_data, mapping_table, feature_description):
    key = "looncomponent_extern_nummer"
    fieldnames = mapping_table[0].split(',')
    dict_tmp = {}
    dict_tmp_field = ['hours_days_km', 'costs', 'is_fixed', 'is_variable', 'declaration', 'allowance', 'travel_related',
                      'company_car', 'overtime', 'leaves', 'health', 'insurance', 'pension', 'overig']
    data = tf.io.parse_single_example(input_data, feature_description)
    idx = fieldnames.index(key)
    lc_number = data[key].numpy().decode("utf-8")

    for i in range(1, len(mapping_table)):
        mapping_datarow = mapping_table[i].split(',')
        if str(mapping_datarow[idx]) == str(lc_number):
            for n, field in enumerate(fieldnames):
                if field not in ['looncomponent_extern_nummer', 'looncomponent', 'Rulebased']:
                    dict_tmp[field] = mapping_datarow[n]

    if dict_tmp == {}:
        for field in dict_tmp_field:
            data[field] = tf.constant(0)
    else:
        for field in dict_tmp_field:
            if dict_tmp[field] in [1, '1']:
                val = 1
            else:
                val = 0
            data[field] = tf.constant(val)

    feature = {}
    for key, value in data.items():
        if value.dtype == 'float32':
            feature[key] = tf.train.Feature(float_list=tf.train.FloatList(value=[value.numpy()]))
        elif value.dtype in ['int64', 'int32']:
            feature[key] = tf.train.Feature(int64_list=tf.train.Int64List(value=[value.numpy()]))
        else:
            feature[key] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.numpy()]))
    return tf.train.Example(features=tf.train.Features(feature=feature))


class Executor(base_executor.BaseExecutor):

    def Do(self, input_dict, output_dict, exec_properties):

        self._log_startup(input_dict, output_dict, exec_properties)

        mapping_uri = exec_properties['mapping_file_path']
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

        mapping_file = os.path.join(mapping_uri, 'grouping_strategy.csv')
        with self._make_beam_pipeline() as pipeline:
            mapping_table = pipeline | beam.io.ReadFromText(mapping_file)
            train_data = (
                    pipeline
                    | 'ReadData' >> beam.io.ReadFromTFRecord(file_pattern=io_utils.all_files_pattern(input_examples_uri))
                    | 'Mapping Wage Components' >> beam.Map(wcmapping, beam.pvalue.AsList(mapping_table), feature_description = feature_description)
                    | 'SerializeExample' >> beam.Map(lambda x: x.SerializeToString())
                    | 'WriteAugmentedData' >> beam.io.WriteToTFRecord(
                                                os.path.join(train_output_examples_uri, "wagecomponent_embedded_data"),
                                                file_name_suffix='.gz')
                    )

        with self._make_beam_pipeline() as pipeline:
            mapping_table = pipeline | beam.io.ReadFromText(mapping_file)
            eval_data = (
                    pipeline
                    | 'ReadData' >> beam.io.ReadFromTFRecord(file_pattern=io_utils.all_files_pattern(eval_input_examples_uri))
                    | 'Mapping Wage Components' >> beam.Map(wcmapping, beam.pvalue.AsList(mapping_table), feature_description = feature_description)
                    | 'SerializeExample' >> beam.Map(lambda x: x.SerializeToString())
                    | 'WriteAugmentedData' >> beam.io.WriteToTFRecord(
                                                os.path.join(eval_output_examples_uri, "wagecomponent_embedded_data"),
                                                file_name_suffix='.gz')
                    )