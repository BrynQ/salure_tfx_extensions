from tfx.components.base import base_executor


def parse_pb(pb, quantile_key):
    pb = tf.train.Example.FromString(pb)

    for f, v in pb.features.ListFields():
        (key, val) = ('', 0)
        for kk, vv in v.items():
            if kk == quantile_key:
                for kkk, vvv in vv.ListFields():
                    if len(vvv.value) == 0:
                        val = ''
                    elif type(vvv.value[0]) == bytes:
                        val = vvv.value[0].decode("utf-8")
                    else:
                        val = vvv.value[0]

    return val

class Executor(base_executor.BaseExecutor):

    def Do(self, input_dict, output_dict, exec_properties):
        self._log_startup(input_dict, output_dict, exec_properties)
        num_quantiles = int(exec_properties['num_quantiles']) + 1
        quantile_key = exec_properties['quantile_key']
        input_examples_uri = artifact_utils.get_split_uri(input_dict['input_data'], 'train')
        output_examples_uri = artifact_utils.get_single_uri(output_dict['percentile_values'])

        with beam.Pipeline() as pipeline:
            train_data = (
                    pipeline
                    | 'ReadData' >> beam.io.ReadFromTFRecord(
                file_pattern=io_utils.all_files_pattern(input_examples_uri))
                    | 'ParseKey' >> beam.Map(parse_pb, quantile_key)
            )
            quantiles = (train_data | 'Quantiles globally' >> beam.transforms.stats.ApproximateQuantiles.Globally(
                num_quantiles=num_quantiles)
                         | 'WriteToFile' >> beam.io.WriteToText(file_path_prefix=output_examples_uri + '/percentile_values',
                                                          shard_name_template='',
                                                          file_name_suffix='.txt')
                         )