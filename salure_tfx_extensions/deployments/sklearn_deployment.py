from string import Template
from typing import Optional, List, Any

from kfp import dsl

from salure_tfx_extensions.deployments.base_deployment import BaseDeployment


class SKLearnDeployment(BaseDeployment):

    # TODO: ALLOW FOR DATAFRAMES USING BYTES AS PROTOCOL

    TEMPLATE = Template("""
{
    "apiVersion": "machinelearning.seldon.io/v1alpha2",
    "kind": "SeldonDeployment",
    "metadata": {
        "name": "$deployment_name"
    },
    "spec": {
        "name": "$deployment_name",
        "predictors": [
            {
                "graph": {
                    "children": [],
                    "implementation": "SKLEARN_SERVER",
                    "modelUri": "pvc://$pvc_name/$model_location",
                    "name": "$deployment_name"
                },
                "name": "$deployment_name",
                "replicas": 1
            }
        ]
    }
}
""")

    def __init__(self,
                 deployment_name: str,
                 pvc_name: str,
                 dependents: List[Any],  # TODO: identify 'Any'
                 model_location: Optional[str] = None):
        self.deployment_name = deployment_name
        self.pvc_name = pvc_name
        self.model_location = model_location or 'data'
        self._dependents = dependents

        seldon_deployment = SKLearnDeployment.TEMPLATE.substitute(
            deployment_name=deployment_name,
            pvc_name=pvc_name,
            model_location=model_location
        )

        self._resource_op = dsl.ResourceOp(
            name=deployment_name,
            action='apply',
            k8s_resource=seldon_deployment,
            success_condition='status.state == Available'
        )  #.after(*dependents)

    @property
    def resource_op(self):
        return self._resource_op

    @property
    def dependents(self):
        return self._dependents


# $deployment_name
# $pvc_name
# $model_location

# deployment_name,
# pvc_name,  [Goes together with model_location to make modelUri]
# model_location [DEFAULT = related to deployment_name]
# request_signature:
#   - bytes
#   -
