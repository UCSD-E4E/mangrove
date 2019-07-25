#!/usr/bin/env bash

docker run --it --runtime=nvidia -v models/:/models \
-v output/:/output \
-v /datadrive/dataset/:/dataset \
cnn-features bash