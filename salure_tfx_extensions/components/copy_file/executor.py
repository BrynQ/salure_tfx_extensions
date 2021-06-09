from distutils.dir_util import copy_tree
from tfx.components.base import base_executor

class Executor(base_executor.BaseExecutor):

    def Do(self, input_dict, output_dict, exec_properties):
        self._log_startup(input_dict, output_dict, exec_properties)
        input_uri = exec_properties['input_path']
        output_uri = exec_properties['output_path']
        copy_tree(input_uri, output_uri)