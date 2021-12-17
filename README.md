[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

**How to run**

1. First install ```sudo apt-get install python3-pip libpcap-dev```
2. Compile the watcher on your machine: ```gcc watcher.c -lpcap```
3. Run the watcher passing through PIPE packets output (pay attention to the interface name): ```sudo ./a.out enp2s0 | python3.7 packet_vision.py```

- Be aware that a lot of images will be created on script directory.


**AI Web API**

## Installing:
* Install PyTorch: ```pip3 install torch```
* Install TorchVision: ```pip3 install torchvision```
* Install SeaBorn: ```pip3 install seaborn```
* Install SkLearn: ```pip3 install sklearn```
* Install MatplotLib: ```pip3 install matplotlib```
* Install flask: ```pip3 install flask```
* Install werkzeug: ```pip3 install werkzeug```

- Testing API: ```curl --form "image=@/home/rodrigo/Desktop/test-image.png" http://52.146.40.114:8080/aioracle```

Thats all folks!


## Embeding CNN models on Low-cost devices Experiments:

* ```pip3 install psutil```
* ```pip3 install Pillow```
* ```pip3 install tensorflow```
* ```pip3 install torch```

* ```./runner.sh```
