from string import Template
from typing import Optional, List, Any
import json

from kfp import dsl

from salure_tfx_extensions.deployments.base_deployment import BaseDeployment
from tfx.components.base.base_component import BaseComponent


class TensorflowDeployment(BaseDeployment):

    # TODO: Figure out signature name

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
                    "implementation": "TENSORFLOW_SERVER",
                    "modelUri": "pvc://$pvc_name/$model_location",
                    "name": "$deployment_name",
                    "parameters": [
                        {
                            "name": "signature_name",
                            "type": "STRING",
                            "value": "$signature_name"
                        },
                        {
                            "name": "model_name",
                            "type": "STRING",
                            "value": "$model_name"
                        }
                    ]
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
                 signature_name: str,
                 model_name: str,
                 dependents: Optional[List[BaseComponent]] = None,
                 model_location: Optional[str] = None):
        self.deployment_name = deployment_name
        self.pvc_name = pvc_name
        self.signature_name = signature_name
        self.model_name = model_name
        self.model_location = model_location or 'data'
        self._dependents = dependents

        self._deployment = TensorflowDeployment.TEMPLATE.substitute(
            deployment_name=deployment_name,
            pvc_name=pvc_name,
            signature_name=signature_name,
            model_name=model_name,
            model_location=model_location,
        )

    # @property
    def resource_op(self) -> dsl.ResourceOp:
        """Generates a kfp dsl ResourceOp.
        Must be initialized in the runtime of the function passed to the kfp compiler
        for the compiler to register it."""
        return dsl.ResourceOp(
            name=self.deployment_name,
            action='apply',
            k8s_resource=json.loads(self._deployment),
            success_condition='status.state == Available'
        )

    @property
    def dependents(self) -> Optional[List[BaseComponent]]:
        return self._dependents
