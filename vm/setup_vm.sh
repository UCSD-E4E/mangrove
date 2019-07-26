# NOTE: MUST RUN WITH SUDO

# for data
sudo apt install unzip
mkdir output
# curl -O INSERT_LINK_TO_DATA [TODO]
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1qL65kT02iaYb4BQPzbtGjKpIoRwscoa5' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1qL65kT02iaYb4BQPzbtGjKpIoRwscoa5" -O input.zip && rm -rf /tmp/cookies.txt
unzip input.zip # input data directory zip file (creates /input folder)
rm input.zip

# latest Nvidia drivers
CUDA_REPO_PKG=cuda-repo-ubuntu1604_10.0.130-1_amd64.deb
wget -O /tmp/${CUDA_REPO_PKG} http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/${CUDA_REPO_PKG} 
sudo dpkg -i /tmp/${CUDA_REPO_PKG}
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub 
rm -f /tmp/${CUDA_REPO_PKG}
sudo apt-get update
sudo apt-get -y install cuda-drivers

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

# grabbing script to run docker container
curl -O https://raw.githubusercontent.com/UCSD-E4E/mangrove/master/vm/docker_run.sh
