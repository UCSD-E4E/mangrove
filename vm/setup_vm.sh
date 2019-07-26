# NOTE: MUST RUN WITH SUDO

# for data
sudo apt install unzip
mkdir output
# curl -O INSERT_LINK_TO_DATA [TODO]

# latest Nvidia drivers
sudo apt-get install cuda-drivers

# installing the docker repo
sudo apt-get update
sudo apt-get -y install \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg-agent \
    software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"

# installing docker engine - community
sudo apt-get update
sudo apt-get -y install docker-ce docker-ce-cli containerd.io
 
# installing nvidia-docker dependencies
distribution=$(.etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo teeetc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker 
