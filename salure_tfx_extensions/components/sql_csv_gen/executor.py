"""Custom Component Executor To Fetch SQL data to CSV"""

import tensorflow as tf
import pymysql
import os
import json
import pandas as pd
from google.protobuf import json_format
from salure_tfx_extensions.proto import mysql_config_pb2
from typing import Any, Dict, List, Text
import apache_beam as beam
from tfx import types
from tfx.components.example_gen import utils
from tfx.dsl.components.base import base_executor
from tfx.proto import example_gen_pb2
from tfx.types import artifact_utils
from tfx.utils import proto_utils
from io import StringIO

_DEFAULT_ENCODING = 'utf-8'


# @beam.typehints.with_input_types(Text)
@beam.typehints.with_output_types(beam.typehints.Iterable[Any])
class _ReadMySQLDoFn(beam.DoFn):

    def __init__(self,
                 mysql_config: mysql_config_pb2.MySQLConnConfig,
                 output_uri):
        super(_ReadMySQLDoFn, self).__init__()
        self.mysql_config = json_format.MessageToDict(mysql_config)
        self.output_uri = output_uri

    def process(self, query: Text):
        client = pymysql.connect(**self.mysql_config)
        cursor = client.cursor()
        cursor.execute(query)

        rows = cursor.fetchall()
        df_uploaded_file = pd.DataFrame()
        if rows:
            try:
                for row in rows:
                    if 'xlsx' in row[3]:  # if this is xlsx format:
                        file_content = json.loads(row[1])
                        file_content = file_content[list(file_content.keys())[0]]
                    else:
                        file_content = row[1]
                    uploaded_files_temp = pd.read_csv(StringIO(file_content),
                                                      engine='python',
                                                      sep=None,
                                                      decimal=',')
                    uploaded_files_temp = uploaded_files_temp[["Medewerkerscode", "Componentcode", "Waarde"]]
                    uploaded_files_temp = uploaded_files_temp.drop(
                        uploaded_files_temp[uploaded_files_temp['Medewerkerscode'].isna()].index)
                    uploaded_files_temp['payment_date'] = row[2]
                    df_uploaded_file = pd.concat([df_uploaded_file, uploaded_files_temp], sort=True)

                df_uploaded_file['boekjaar'] = df_uploaded_file['payment_date'].dt.year
                df_uploaded_file['periode_uitgevoerd'] = df_uploaded_file['payment_date'].dt.month
                df_uploaded_file = df_uploaded_file[
                    ["Medewerkerscode", "Componentcode", "Waarde", "boekjaar", "periode_uitgevoerd"]]
                df_uploaded_file = df_uploaded_file.rename(columns={"Medewerkerscode": "medewerker_id",
                                                                    "Componentcode": "looncomponent_extern_nummer",
                                                                    "Waarde": "adjusted_amount"})
                df_uploaded_file.looncomponent_extern_nummer = df_uploaded_file.looncomponent_extern_nummer.astype(
                    "str")
                df_uploaded_file.medewerker_id = df_uploaded_file.medewerker_id.astype("str")
                df_uploaded_file = df_uploaded_file.groupby(['medewerker_id', 'looncomponent_extern_nummer',
                                                             'boekjaar', 'periode_uitgevoerd'])[
                    'adjusted_amount'].sum().reset_index()
                df_uploaded_file.to_csv(os.path.join(self.output_uri, 'uploaded_files.csv'),
                                        index=False,
                                        sep=',',
                                        header=True)

            finally:
                cursor.close()
                client.close()


class Executor(base_executor.BaseExecutor):

    def Do(self,
           input_dict: Dict[Text, List[types.Artifact]],
           output_dict: Dict[Text, List[types.Artifact]],
           exec_properties: Dict[Text, Any]) -> None:
        self._log_startup(input_dict, output_dict, exec_properties)
        input_config = example_gen_pb2.Input()
        proto_utils.json_to_proto(exec_properties[utils.INPUT_CONFIG_KEY], input_config)
        output_config = example_gen_pb2.Output()
        proto_utils.json_to_proto(exec_properties[utils.OUTPUT_CONFIG_KEY], output_config)

        examples_artifact = artifact_utils.get_single_instance(output_dict[utils.EXAMPLES_KEY])
        examples_artifact.split_names = artifact_utils.encode_split_names(
            utils.generate_output_split_names(input_config, output_config))

        conn_config = example_gen_pb2.CustomConfig()
        json_format.Parse(exec_properties['custom_config'], conn_config)
        mysql_config = mysql_config_pb2.MySQLConnConfig()
        conn_config.custom_config.Unpack(mysql_config)

        output_dict['examples'][0].split_names = (artifact_utils.encode_split_names(["single_split"]))
        output_uri = artifact_utils.get_single_uri(output_dict['examples'])
        query = eval(exec_properties['input_config'])['splits'][0]['pattern']

        with self._make_beam_pipeline() as pipeline:
            example = (pipeline
                       | 'Query' >> beam.Create([query])
                       | 'ReadFromDB' >> beam.ParDo(_ReadMySQLDoFn(mysql_config, output_uri)))