FROM tensorflow/tensorflow:1.13.2-gpu-py3

RUN apt-get install git -y
RUN pip install keras sklearn h5py numpy
RUN mkdir work/
WORKDIR work/

VOLUME /dataset
VOLUME /output

ADD extract.py .
ADD classify.py .

CMD ["/bin/bash"]g