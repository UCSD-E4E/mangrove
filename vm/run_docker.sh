docker pull awmaucsd/retrain

echo "Note: GPUs are typicall indexed starting at 1, i.e. 2 GPUS will be 0,1"
read -N 1 -s -p "Which GPU: " gpu_num

docker run -it --name cnn_app -u=0 --gpus "device=$gpu_num" \
    -v $(pwd)/input:/mnt/input \
    -v $(pwd)/output:/mnt/output \
    awmaucsd/retrain
