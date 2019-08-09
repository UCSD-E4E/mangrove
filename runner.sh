#!/usr/bin/env bash

docker run -it -v "$(pwd)/psc3_4/":/output \
-v "/features-2/psc_site_3-4_tiles_jpg":/dataset \
--runtime=nvidia --user 1000:1000 \
features python extract.py \
-i=/dataset -o=/output \
-uf -n=60 -b=512