# I-SIM
A simulator that is built upon the INTERACTION Dataset. We will release it as soon as we finish reorganizing and cleaning the codes. An early version is provided in our previous work, [TrajGen][website_trajgen].

[website_trajgen]: https://github.com/gaoyinfeng/TrajGen/

## Kinematic Model Validation
To validate that the dynamic bicycle model and the PID controllers we use in the simulator are enough for simulating vehicles' motions and controlling vehicles given high-level speed control, we conducted several tests. 

**Left**: Bicycle model validation. We use its inverse model to generate control signals based on the ego vehicle's log trajectory, then the generated control signals are used to control the ego vehicle in turn. As the video shows, the ego vehicle (red) and the white ghost vehicle (represents the log trajectory) almost overlap with each other, which means the position errors are very small. Therefore, this bicycle model is reasonable for showing the realistic motions of the vehicles.

**Right**: PID controllers validation. To test the PID controllers, we made an extreme situation, i.e. using the maximum speed target as the PID controllers input, and see whether the PID can keep the vehicle in its tracks. We show some typical difficult cases, where a large vehicle drives on a continuously curved route, and a regular vehicle drives to finish a U-turn. The results are good enough to prove the rationality of PID controllers.

<img width="45%" src="https://github.com/gaoyinfeng/I-SIM/blob/main/pics/Bicycle Model Validation.gif"> <img width="45%" src="https://github.com/gaoyinfeng/I-SIM/blob/main/pics/PID Controllers Validation.gif">


## Acknowledgement
We appreciate the following github repos for their valuable code base or dataset:

https://github.com/fzi-forschungszentrum-informatik/Lanelet2

https://github.com/interaction-dataset/interaction-dataset
