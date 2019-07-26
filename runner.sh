#!/usr/bin/env bash

docker run -it -v "$(pwd)/output/":/output \
-v "/features-2/train":/dataset/train \
--runtime=nvidia --user 1000:1000 \
features python extract.py