#!/usr/bin/env bash

mkdir -p "$1"

docker run -it -v "$(pwd)/$1/":/output \
-v "$2":/dataset \
--runtime=nvidia --user 1000:1000 \
features python extract.py \
-i=/dataset -o=/output \
-n=400 -b=512 -s=128