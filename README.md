# Salure TFX Extensions

Repository: https://bitbucket.org/salurebi/salure_tfx_extensions/src/master/

Docker Image over at hub.docker.com: salure/kubeflow_tfx_extensions


## Goal
Two goals:

- To add functionality as required by internal demand of Salure.
- To provide helper functions and new components for TFX


## Components

### MySQLExampleGen
Needs a salure_tfx_extensions.proto.mysql_config_pb2.MySQLConnConfig, and an SQL query.

The component code, and parameters:
https://bitbucket.org/salurebi/salure_tfx_extensions/src/master/salure_tfx_extensions/components/mysql_example_gen/component.py

### ModelValidator
Just like the tfx ModelValidator, but (for now) will always pass
