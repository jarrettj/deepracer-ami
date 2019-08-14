# deepracer-ami
AWS DeepRacer AMI

# Intro
My attempt at using all the knowledge gained so far from the community. [Join here](https://join.slack.com/t/deepracer-community/shared_invite/enQtNjgyMTQ5MjA0OTMzLTZjN2E5N2MyMGIxYTYyNGRiNGU3NDhmNjBhMTlkZmY5ZDhmMjBmY2I1YjUzNTRkMjZjZDVlNDE1MTAyOWNiYjE). Will try and make setting up a cheaper environment on EC2 easier. 

Uses crr0004 [repo](https://github.com/crr0004/deepracer). And followed this guide as well of jonathantse [here](https://medium.com/@jonathantse/train-deepracer-model-locally-with-gpu-support-29cce0bdb0f9). Also do read crr0004's [wiki page](https://github.com/crr0004/deepracer/wiki), it's very helpful.

# EC2 Launch
[Detailed description](https://github.com/jarrettj/deepracer-ami/wiki/Launch-EC2-instance) with screenshots.

## g2.2xlarge
|vCPU|ECU|Memory (GiB)|Instance Storage (GB)|Linux/UNIX Usage|Spot Usage
| ---- | ---- | ---- | ---- | ---- | ---- |
|8|26|15 GiB 1|60 SSD|$0.65 per Hour|$0.225|

[GPU Details]
nVidia Corporation GK104GL [GRID K520](https://www.techpowerup.com/gpu-specs/grid-k520.c2312)

Real Time Factor of 0.7 - 0.9.
Sample of time to train 10 models on the AWS_track with all settings left as default:
```
ls /mnt/data/minio/bucket/rl-deepracer-sagemaker/model/*.pb -laht
-rw-r--r-- 1 ubuntu ubuntu 23M Aug  2 12:38 /mnt/data/minio/bucket/rl-deepracer-sagemaker/model/model_9.pb
...
-rw-r--r-- 1 ubuntu ubuntu 23M Aug  2 12:11 /mnt/data/minio/bucket/rl-deepracer-sagemaker/model/model_0.pb
```

Took about 30 minutes for the initial test. But stopping the training and resuming with only a slight change in action space speed and the reward function made the next 10 models generate in nearly 4 hours.

Notes:
Ran into OOM issue though during extended periods of training. I'll retest when there's another track update or the like.
Really strugging with this instance type. The CPUs are awesome, but the GPU is not fully utilised for some reason. Will try and get some answers from AWS. As the training goes on the GPU lets it down.

## g3s.xlarge
|vCPU|ECU|Memory (GiB)|Instance Storage (GB)|Linux/UNIX Usage|Spot Usage
| ---- | ---- | ---- | ---- | ---- | ---- |
|4|13|30.5 GiB|EBS Only|$0.75 per Hour|$0.3|

[GPU Details]
nVidia Corporation [Tesla M60](https://www.nvidia.com/object/tesla-m60.html)

Real Time Factor of 0.5 - 0.7. This is concerning. As it affects training negatively. As it will take longer to train.
Sample of time to train 10 models on the AWS_track with all settings left as default:
Took about 80 minutes as well.

## g3.4xlarge
|vCPU|ECU|Memory (GiB)|Instance Storage (GB)|Linux/UNIX Usage|Spot Usage
| ---- | ---- | ---- | ---- | ---- | ---- |
|16|47|122 GiB|EBS Only|$1.14 per Hour|$0.5|

[GPU Details] 
nVidia Corporation [Tesla M60](https://www.nvidia.com/object/tesla-m60.html)

Real Time Factor of 0.9 - 1.
Sample of time to train 10 models on the AWS_track with all settings left as default:
```
ls /mnt/data/minio/bucket/rl-deepracer-sagemaker/model/*.pb -laht
-rw-r--r-- 1 ubuntu ubuntu 23M Aug  7 10:10 /mnt/data/minio/bucket/rl-deepracer-sagemaker/model/model_9.pb
...
-rw-r--r-- 1 ubuntu ubuntu 23M Aug  7 09:46 /mnt/data/minio/bucket/rl-deepracer-sagemaker/model/model_0.pb
```

Took about 24 minutes for the initial test. 

When I continued training on an existing model it took a bit longer. But not as long as the g2.2xlarge.
```
ls /mnt/data/minio/bucket/rl-deepracer-sagemaker/model/*.pb -laht
-rw-r--r-- 1 ubuntu ubuntu 23M Aug  8 07:50 /mnt/data/minio/bucket/rl-deepracer-sagemaker/model/model_9.pb
...
-rw-r--r-- 1 ubuntu ubuntu 23M Aug  8 06:39 /mnt/data/minio/bucket/rl-deepracer-sagemaker/model/model_0.pb
```

Took about 70 minutes for the pretrained test. 

I made the same changes as before. Increased the action space speed and then updated to reward function slightly to cater for action space speed update.

# Changes you have to make
In robomaker.env and rl_coach/env.sh
1. WORLD_NAME - Replace with track you want to train
2. S3_ENDPOINT_URL = service mini status - Replace with minio endpoint

In rl_coach/rl_deepracer_coach_robomaker.py
1. instance_type - local or local_gpu depending on ec2 type

# Starting training
You don't have to use nohup, I simply use it to background the process and capture output to a file for analysis later.
Sagemaker:
```
nohup python rl_deepracer_coach_robomaker.py > sagemaker.log &
```
Without nohup:
```
python rl_deepracer_coach_robomaker.py
```

Robomaker:
```
nohup docker run --rm --name dr --env-file ./robomaker.env --network sagemaker-local -p 8080:5900 -i crr0004/deepracer_robomaker:console > robomaker.log &
```
Without nohup:
```
docker run --rm --name dr --env-file ./robomaker.env --network sagemaker-local -p 8080:5900 -it crr0004/deepracer_robomaker:console
```

# Evaluation
Update WORLD_NAME and the NUMBER_OF_TRIALS you want.
```
docker run --rm --name dr_e --env-file ./robomaker.env --network sagemaker-local -p 8181:5900 -it -e "WORLD_NAME=reinvent_base" -e "NUMBER_OF_TRIALS=1" -e "METRICS_S3_OBJECT_KEY=custom_files/eval_metric.json" crr0004/deepracer_robomaker:console "./run.sh build evaluation.launch"
```

# Log analysis
Use port forwarding:
```
ssh -i ~/path/to/pem -L 8888:localhost:8888 ubuntu@host_ip
```

Once logged in, run the following on the EC2 instance:
```
cd /mnt/data/aws-deepracer-workshops/log-analysis
jupyter notebook
```

Now on your machine open the link that the above command has generated. Should be something similar to:
```
http://localhost:8888/?token=07fa15f8d78e5a137c4aef2dc4556c06d7040b926bc5fcf0
```

I've put the aws-deepracer-workshops repo on the data mount at /mnt/data. You should have access to the DeepRacer Log Analysis.ipynb. Documented [here](https://codelikeamother.uk/using-jupyter-notebook-for-analysing-deepracer-s-logs).

All you have to do now is copy your robomaker.log file into the /mnt/data/aws-deepracer-workshops/log-analysis/logs folder, then reference it in DeepRacer Log Analysis. I replace the fname with the path to the generated robomaker.log file.

# Optimise GPU 
This might be needed. But first run your training before setting these. And remember it might take a while for the training to start. If the GPU is in use you should see the following:
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 390.116                Driver Version: 390.116                   |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GRID K520           Off  | 00000000:00:03.0 Off |                  N/A |
| N/A   36C    P8    17W / 125W |     48MiB /  4037MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|    0      3004      C   /usr/bin/python                               37MiB |
+-----------------------------------------------------------------------------+
```

If you are using a GPU based image, run the following: ([Optimize GPU](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/optimize_gpu.html))
```
sudo nvidia-persistenced
sudo nvidia-smi --auto-boost-default=0
```

Check the clock speeds
```
nvidia-smi -q -d SUPPORTED_CLOCKS
```
Use the two highest memory values in the following command
```
sudo nvidia-smi -ac 2505,1177
```

# Issues
## China track
Add the following to RoboMaker startup:
```
-v {path_to_your_project_folder}/simulation/aws-robomaker-sample-application-deepracer/simulation_ws/src:/app/robomaker-deepracer/simulation_ws/src

nohup docker run -v /home/ubuntu/deepracer/simulation/aws-robomaker-sample-application-deepracer/simulation_ws/src:/app/robomaker-deepracer/simulation_ws/src --rm --name dr --env-file ./robomaker.env --network sagemaker-local -p 8080:5900 -i crr0004/deepracer_robomaker:console > robomaker.log &
```

## OOM errors when using GPU
Create folder sagemaker-redis-fix
```
mkdir sagemaker-redis-fix
cd sagemaker-redis-fix
```

Create Dockerfile:
```
Dockerfile
FROM crr0004/sagemaker-rl-tensorflow:nvidia

COPY start.sh /usr/local/bin/start.sh
RUN chmod +x /usr/local/bin/start.sh

# Starts framework
ENTRYPOINT ["/bin/bash", "-c", "start.sh train"]
CMD ["start.sh", "train"]
```
Create new start.sh
```
#!/usr/bin/env bash

echo "\$1 is $1"
if [ "$1" == 'train' ]
then
	echo "In train start.sh"
    # Remove all nvidia gl libraries if they exists to run training in SageMaker.
    rm -rf /usr/local/nvidia/lib/libGL*
    rm -rf /usr/local/nvidia/lib/libEGL*
    rm -rf /usr/local/nvidia/lib/libOpenGL*
    rm -rf /usr/local/nvidia/lib64/libGL*
    rm -rf /usr/local/nvidia/lib64/libEGL*
    rm -rf /usr/local/nvidia/lib64/libOpenGL*

    CURRENT_HOST=$(jq .current_host  /opt/ml/input/config/resourceconfig.json)
	echo "Current host is $CURRENT_HOST"

    sed -ie "s/PLACEHOLDER_HOSTNAME/$CURRENT_HOST/g" /changehostname.c

	echo "Compiling changehostname.c"
    gcc -o /changehostname.o -c -fPIC -Wall /changehostname.c
    gcc -o /libchangehostname.so -shared -export-dynamic /changehostname.o -ldl

	echo "Done Compiling changehostname.c"
	export XAUTHORITY=/root/.Xauthority
	export DISPLAY=:0 # Select screen 0 by default.
	xvfb-run -f $XAUTHORITY -l -n 0 -s ":0 -screen 0 1400x900x24" jwm &
	CUDA_VISIBLE_DEVICES=-1 redis-server --bind 0.0.0.0 &
	x11vnc -bg -forever -nopw -rfbport 5800 -display WAIT$DISPLAY &
    LD_PRELOAD=/libchangehostname.so train &
	wait
elif [ "$1" == 'serve' ]
then
    serve
fi
```

The CUDA_VISIBLE_DEVICES=-1 tells redis to not use the GPU.

Build your own image to use:
```
docker build -t sagemaker-redis-fix .
```

Update rl_coach/rl_deepracer_coach_robomaker.py with your own image:
```
image_name="sagemaker-redis-fix",
```

Or use mine jljordaan/sagemaker-rl-tensorflow:nvidia. 

go into rl_coach/rl_deepracer_coach_robomaker.py

find the place where you select sagemaker image, starting with crr0004/

replace crr0004/ with jljordaan/

This has worked for me, but it has not worked for others. Will have to look into this issue in more detail when I have a chance. 

# TODO: 
1. Script it all together to easily manage spot instances stopping.
2. Figure out why it runs on certain GPU instances only.
3. I'm working on a Mac. Not sure what issues would be encountered elsewhere. 



