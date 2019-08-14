FROM tensorflow/tensorflow:1.13.2-gpu-py3

RUN pip install keras sklearn h5py numpy pillow
RUN mkdir work/
WORKDIR work/

VOLUME /dataset
VOLUME /output

ADD extract.py .
ADD classify.py .
ADD ../vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5 .
RUN export KERAS_HOME="$(pwd)"

CMD ["/bin/bash"]