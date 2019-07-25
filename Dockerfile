FROM tensorflow/tensorflow:latest-gpu

RUN apt-get install git -y
RUN pip install keras sklearn h5py
RUN mkdir work/
WORKDIR work/

VOLUME /dataset
VOLUME /output
VOLUME /models

ADD extract.py .
