echo "Note: GPUs are typicall indexed starting at 1, i.e. 2 GPUS will be 0,1"
read -p "Which GPU: " gpu_num

docker run -it --gpus '"device=$gpu_num"' \
    -v $(pwd)/input:/input \
    -v $(pwd)/output:/output \
    awmaucsd/retrain
