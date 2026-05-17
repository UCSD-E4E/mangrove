docker pull awmaucsd/retrain

read -p "Give the docker container a name: " container_name

echo "Note: GPUs are typically indexed starting at 1, i.e. 2 GPUS will be 0,1"
read -p "Which GPU: " gpu_num

docker run -it --name $container_name -u=0 --gpus "device=$gpu_num" \
    -v $(pwd)/input:/mnt/input \
    -v $(pwd)/output:/mnt/output \
    awmaucsd/retrain
