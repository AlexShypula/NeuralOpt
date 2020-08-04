#!/bin/bash
sudo docker pull stanfordpl/stoke:latest
sudo docker run -d -P --name TorchStoke -p 6000:6000 -v $(pwd)/docker:/home/stoke/docker -it stanfordpl/stoke:latest
#sudo docker run -d -P --name TorchStoke2 -p 32770:22 -p 6000:6000 -v $(pwd)/docker:/home/stoke/docker -it torch_stoke_base 
sudo docker port TorchStoke 22
