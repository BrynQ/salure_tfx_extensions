# UPDATE ON 'VERSION' UPDATE
FROM tensorflow/tfx:1.0.0rc1

RUN pip install scikit-learn==0.23.1
RUN pip install tensorflow==2.3.0

COPY . /salure_tfx_extensions
RUN pip install /salure_tfx_extensions
