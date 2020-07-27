#!/bin/bash
sudo docker pull stanfordpl/stoke:latest
sudo docker run -d -P --name TorchStoke -v $(pwd)/docker:/home/stoke/docker -it stanfordpl/stoke:latest
sudo docker port TorchStoke 22
