FROM nvidia/cuda

VOLUME /output
VOLUME /input

# Installing basic Linux tools
RUN apt-get update && apt-get -y install curl

# Curling tools for CNN retraining
RUN curl -O https://raw.githubusercontent.com/UCSD-E4E/mangrove/master/CNN%20Development/cnn_retrain2.py    
RUN curl -O https://raw.githubusercontent.com/UCSD-E4E/mangrove/master/CNN%20Development/retrain_docker.sh

# Changing permissions on files
RUN chmod +x cnn_retrain2.py retrain_docker.sh

# Installing tf packages
RUN apt update && apt -y install python3-pip && pip3 install tensorflow-gpu && pip3 install tensorflow-hub

# Setting default user for docker container
USER 1000:1000
