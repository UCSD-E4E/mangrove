#!/usr/bin/env bash

docker run --it --runtime=nvidia \
-v output/:/output \
-v /features-2/dataset/:/dataset \
--user "$(id -u):$(id -g)" \
cnn-features