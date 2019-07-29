# copying and chwoning files
docker cp cnn_app:/mnt/output $(pwd)
chown -R 1000:1000 $(pwd)

# cleaning up containers
docker container prune -f
docker container rm cnn_app
