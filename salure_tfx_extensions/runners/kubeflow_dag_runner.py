from kfp import dsl

from typing import Optional, List
import os

from tfx.orchestration.kubeflow.kubeflow_dag_runner import KubeflowDagRunner
from tfx.orchestration import pipeline as tfx_pipeline
from tfx.orchestration.config import config_utils
from tfx.orchestration.kubeflow import base_component
from salure_tfx_extensions.deployments.base_deployment import BaseDeployment


class KubeflowDagRunner(KubeflowDagRunner):

    def _construct_pipeline_graph_with_deployments(self, pipeline: tfx_pipeline.Pipeline,
                                                   pipeline_root: dsl.PipelineParam,
                                                   deployments: List[BaseDeployment]):
        component_to_kfp_op = {}

        # Assumption: There is a partial ordering of components in the list, i.e.,
        # if component A depends on component B and C, then A appears after B and C
        # in the list.
        for component in pipeline.components:
            # Keep track of the set of upstream dsl.ContainerOps for this component.
            depends_on = set()

            for upstream_component in component.upstream_nodes:
                depends_on.add(component_to_kfp_op[upstream_component])

            (component_launcher_class,
             component_config) = config_utils.find_component_launch_info(
                self._config, component)

            kfp_component = base_component.BaseComponent(
                component=component,
                component_launcher_class=component_launcher_class,
                depends_on=depends_on,
                pipeline=pipeline,
                pipeline_name=pipeline.pipeline_info.pipeline_name,
                pipeline_root=pipeline_root,
                tfx_image=self._config.tfx_image,
                kubeflow_metadata_config=self._config.kubeflow_metadata_config,
                component_config=component_config,
                pod_labels_to_attach=self._pod_labels_to_attach)

            for operator in self._config.pipeline_operator_funcs:
                kfp_component.container_op.apply(operator)

            component_to_kfp_op[component] = kfp_component.container_op

        for deployment in deployments:
            resource_op = deployment.resource_op
            depends_on = set()
            for dependent in deployment.dependents:
                depends_on.add(component_to_kfp_op[dependent])

            for dependent in depends_on:
                resource_op.after(dependent)

    def run(self, pipeline: tfx_pipeline.Pipeline,
            deployments: Optional[List[BaseDeployment]] = None):
        pipeline_root = tfx_pipeline.ROOT_PARAMETER

        dsl_pipeline_root = dsl.PipelineParam(
            name=pipeline_root.name, value=pipeline.pipeline_info.pipeline_root)
        self._params.append(dsl_pipeline_root)

        if deployments is None:
            def _construct_pipeline():
                self._construct_pipeline_graph(pipeline, dsl_pipeline_root)
        else:
            def _construct_pipeline():
                self._construct_pipeline_graph_with_deployments(pipeline, dsl_pipeline_root, deployments)

        self._parse_parameter_from_pipeline(pipeline)

        file_name = self._output_filename or pipeline.pipeline_info.pipeline_name + '.tar.gz'

        self._compiler._create_and_write_workflow(
            pipeline_func=_construct_pipeline,
            pipeline_name=pipeline.pipeline_info.pipeline_name,
            params_list=self._params,
            package_path=os.path.join(self._output_dir, file_name))
