#!/usr/bin/env bash

docker run -it -v "$(pwd)/output-site10/":/output \
-v "/features-2/site10_labels":/dataset \
--runtime=nvidia --user 1000:1000 \
features python extract.py \
-i=/dataset -o=/output \
-uf -n=60 -b=512