import tensorflow as tf
from typing import Any, Dict, Optional, Text, Tuple
from tfx.components.base import base_driver
from tfx.orchestration import data_types


class Driver(base_driver.BaseDriver):
    """Custom driver for model validator."""

    def _fetch_last_blessed_model(
            self,
            component_id: Text,
    ) -> Tuple[Optional[Text], Optional[int]]:
        """Fetch last blessed model in metadata based on span."""
        previous_blessed_models = []
        for a in self._metadata_handler.get_artifacts_by_type('ModelBlessingPath'):
            if (a.custom_properties['blessed'].int_value == 1 and
                    a.custom_properties['component_id'].string_value == component_id):
                previous_blessed_models.append(a)

        if previous_blessed_models:
            # TODO(b/138845899): consider use span instead of id.
            last_blessed_model = max(
                previous_blessed_models, key=lambda artifact: artifact.id)
            return (
                last_blessed_model.custom_properties['current_model'].string_value,
                last_blessed_model.custom_properties['current_model_id'].int_value)
        else:
            return None, None

    def resolve_exec_properties(
            self,
            exec_properties: Dict[Text, Any],
            component_info: data_types.ComponentInfo
    ) -> Dict[Text, Any]:
        """Overrides BaseDriver.resolve_exec_properties()."""
        (exec_properties['blessed_model'],
         exec_properties['blessed_model_id']) = self._fetch_last_blessed_model(
            component_info.component_id)
        exec_properties['component_id'] = component_info.component_id
        tf.logging.info('Resolved last blessed model {}'.format(
            exec_properties['blessed_model']))
        return exec_properties
