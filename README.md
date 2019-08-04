# deepracer-ami
AWS DeepRacer AMI

# Intro
My attempt at using all the knowledge gained so far from the community. [Join here](https://join.slack.com/t/deepracer-community/shared_invite/enQtNjgyMTQ5MjA0OTMzLTZjN2E5N2MyMGIxYTYyNGRiNGU3NDhmNjBhMTlkZmY5ZDhmMjBmY2I1YjUzNTRkMjZjZDVlNDE1MTAyOWNiYjE). Will try and make setting up a cheaper environment on EC2 easier. 

Uses crr0004 [repo](https://github.com/crr0004/deepracer). And followed this guide as well of jonathantse [here](https://medium.com/@jonathantse/train-deepracer-model-locally-with-gpu-support-29cce0bdb0f9). Also do read crr0004's [wiki page](https://github.com/crr0004/deepracer/wiki), it's very helpful.

# Find in AMI Public repo
Search for deepracer-ami in AWS EC2 Console. Available in the US East 1 and EU West 1 regions. Try and choose an EC2 with GPU enabled. 

Top pick for me are the g3 instance class, spots of course.

## g2.2xlarge
|vCPU|ECU|Memory (GiB)|Instance Storage (GB)|Linux/UNIX Usage|Spot Usage
| ---- | ---- | ---- | ---- | ---- | ---- |
|8|26|15 GiB 1|60 SSD|$0.65 per Hour|$0.225

Real Time Factor of 0.7 - 0.9.
Sample of time to train 10 models on the AWS_track with all settings left as default:

```
ls /mnt/data/minio/bucket/rl-deepracer-sagemaker/model/*.pb -laht
-rw-r--r-- 1 ubuntu ubuntu 23M Aug  2 12:41 /mnt/data/minio/bucket/rl-deepracer-sagemaker/model/model_10.pb
-rw-r--r-- 1 ubuntu ubuntu 23M Aug  2 12:39 /mnt/data/minio/bucket/rl-deepracer-sagemaker/model/model_9.pb
-rw-r--r-- 1 ubuntu ubuntu 23M Aug  2 12:38 /mnt/data/minio/bucket/rl-deepracer-sagemaker/model/model_8.pb
-rw-r--r-- 1 ubuntu ubuntu 23M Aug  2 12:36 /mnt/data/minio/bucket/rl-deepracer-sagemaker/model/model_7.pb
-rw-r--r-- 1 ubuntu ubuntu 23M Aug  2 12:33 /mnt/data/minio/bucket/rl-deepracer-sagemaker/model/model_6.pb
-rw-r--r-- 1 ubuntu ubuntu 23M Aug  2 12:31 /mnt/data/minio/bucket/rl-deepracer-sagemaker/model/model_5.pb
-rw-r--r-- 1 ubuntu ubuntu 23M Aug  2 12:29 /mnt/data/minio/bucket/rl-deepracer-sagemaker/model/model_4.pb
-rw-r--r-- 1 ubuntu ubuntu 23M Aug  2 12:25 /mnt/data/minio/bucket/rl-deepracer-sagemaker/model/model_3.pb
-rw-r--r-- 1 ubuntu ubuntu 23M Aug  2 12:23 /mnt/data/minio/bucket/rl-deepracer-sagemaker/model/model_2.pb
-rw-r--r-- 1 ubuntu ubuntu 23M Aug  2 12:18 /mnt/data/minio/bucket/rl-deepracer-sagemaker/model/model_1.pb
-rw-r--r-- 1 ubuntu ubuntu 23M Aug  2 12:11 /mnt/data/minio/bucket/rl-deepracer-sagemaker/model/model_0.pb
```

Took about 30 minutes. 

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

Robomaker:
```
nohup docker run --rm --name dr --env-file ./robomaker.env --network sagemaker-local -p 8080:5900 -i crr0004/deepracer_robomaker:console > robomaker.log &
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

#TODO: 
1. Script it all together to easily manage spot instances stopping.
2. Figure out why it runs on certain GPU instances only.
3. I'm working on a Mac. Not sure what issues would be encountered elsewhere. 
4. Add China track.
