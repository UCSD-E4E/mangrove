read "Enter container name to retrieve output from: " container_name

# copying and chwoning files
docker cp $container_name:/mnt/output $(pwd)
chown -R 1000:1000 $(pwd)

# cleaning up containers
docker container prune -f
docker container rm -f $container_name
