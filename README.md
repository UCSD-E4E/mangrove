# Mangrove Classification Using Machine Learning 

Team members:

Dillon Hicks, Savannah Longoria, Kathy Qi, Catherine Lee

### Instructions on Using Docker 

Build docker image :  
`Docker build –t dockerfile  .`   

List all images: 
`Docker image ls` 

List all containers: 
`Docker ps`

Create docker container 
`Docker create -t - I dockerfile`

Start docker container  
`Docker start -a -I <container id>`

SSH into a running container:
`docker exec -it <container name> /bin/bash`
 
Run docker image with gpu
`nvidia-docker run -i dockerfile`
`docker exec -it <container name> /bin/bash`
