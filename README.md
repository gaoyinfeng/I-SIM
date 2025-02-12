# I-SIM
A simulator that is built upon the INTERACTION Dataset. An early version is provided in our previous work, [TrajGen][website_trajgen].

[website_trajgen]: https://github.com/gaoyinfeng/TrajGen/

## Kinematic Model Validation
To validate that the dynamic bicycle model and the PID controllers we use in the simulator are enough for simulating vehicles' motions and controlling vehicles given high-level speed control, we conducted several tests. 

**Left**: Bicycle model validation. We use its inverse model to generate control signals based on the ego vehicle's log trajectory, then the generated control signals are used to control the ego vehicle in turn. As the video shows, the ego vehicle (red) and the white ghost vehicle (represents the log trajectory) almost overlap with each other, which means the position errors are very small. Therefore, this bicycle model is reasonable for showing the realistic motions of the vehicles.

**Right**: PID controllers validation. To test the PID controllers, we made an extreme situation, i.e. using the maximum speed target as the PID controllers input, and see whether the PID can keep the vehicle in its tracks. We show some typical difficult cases, where a large vehicle drives on a continuously curved route, and a regular vehicle drives to finish a U-turn. The results are good enough to prove the rationality of PID controllers.

<img width="45%" src="https://github.com/gaoyinfeng/I-SIM/blob/main/pics/Bicycle Model Validation.gif"> <img width="45%" src="https://github.com/gaoyinfeng/I-SIM/blob/main/pics/PID Controllers Validation.gif">


## Manual Instructions

To properly run I-ISM on your system, you should clone this repository, and follow the instruction below to install the dependencies for I-SIM.


### Dependencies for I-SIM simulator

Since the HD maps of Interaction Dataset uses the format of Lanelet2, you need to build Lanelet2 Docker first, we provide a modified version of [Lanelet2](https://github.com/fzi-forschungszentrum-informatik/Lanelet2) in our repo:

```shell
cd Lanelet2
docker build -t #image_name# .
```

After build, run docker container and do port mapping, we use port 5557-5561 as example:

```shell
docker run -it -e DISPLAY -p 5557-5561:5557-5561 -v $path for ISIM$:/home/developer/workspace/interaction-dataset-master -v /tmp/.X11-unix:/tmp/.X11-unix --user="$(id --user):$(id --group)" --name #container_name# #image_name#:latest bash
```

Update dependencies:

```shell
sudo apt update
sudo apt install python-tk
```

Restart the computer and restart the container：

```shell
docker restart #container_name#
docker exec -it #container_name# bash
cd interaction_gym/
export DISPLAY=:0
```


## Usage

### Qucik Start

To train or test your agent, you have to run I-SIM manually, we assume that I-SIM runs at port 5557 inside a docker container:

```shell
docker exec -it isim57 bash
cd interaction_gym/
python interaction_env.py --port=5557
```


## Acknowledgement

We appreciate the following github repos for their valuable code base or dataset:

https://github.com/fzi-forschungszentrum-informatik/Lanelet2

https://github.com/interaction-dataset/interaction-dataset
