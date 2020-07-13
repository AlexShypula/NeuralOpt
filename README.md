# NeuralOpt

Directions

To obtain the docker container in order to run [STOKE](https://github.com/StanfordPL/stoke) run the docker_setup.sh script which will pull the right docker container for the project, run it, and mount the local `/docker` folder to the `/home/stoke/docker` folder in the conainer. 

Afterwards, you'll have to install stoke inside the container. 

Run `docker ps` and under `PORTS` you'll see the port that is exposed on the container for ssh `0.0.0.0.xxxxx->22/tcp`. Then run `ssh -p xxxx stoke@127.0.0.1` and enter in the password `stoke`. After you ssh into the container, install and test stoke by running the following. 


```
cd stoke
./configure.sh
make
make test
```

Make and make test may take while to run properly. If everything goes right you'll be able to then run the scripts for training where the `/docker` folder will a volume through which data can be shared between the training script and the container which will evaluate output saved in the `/docker` folder. 
