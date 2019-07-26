FROM tensorflow/tensorflow:latest-gpu

RUN apt-get install git -y
RUN pip install keras sklearn h5py numpy
WORKDIR cnn-features/

VOLUME /dataset
VOLUME /output

ADD extract.py .
ADD classify.py .

CMD ["/bin/bash"]