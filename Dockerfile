# UPDATE ON 'VERSION' UPDATE
FROM tensorflow/tfx:0.27.0

RUN pip install scikit-learn==0.23.1

COPY . /salure_tfx_extensions
RUN pip install /salure_tfx_extensions