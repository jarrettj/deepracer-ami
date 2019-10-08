#!/usr/bin/ruby
require 'date'

$userHome = "/home/jcmentoor"
$deepracerRepo = "#$userHome/deepracer"
$sagemakerImage = "crr0004/sagemaker-rl-tensorflow:console_v1.1"
$robomakerImage = "crr0004/deepracer_robomaker:console_v1.1"
$instanceType = "local"
$dataPath = "/mnt/data"
$modelPath = "#$dataPath/minio/bucket"
$modelS3Prefix = "rl-deepracer-sagemaker"
$s3Backup = "s3://aws-deepracer-c54751c5-fede-4e6f-9c4e-70d9ba19a191/bucket"
$pythonExe = "python"
$minioIP = "http://172.17.0.1:9000"
$isMac = false
$offset = 1
$versions = 10
$source = "."
$evalCount = 10
$fileTime = 30
$waitTime = 30
$modelCount = 20
$driver = "CanadaLocalV1"
$train = true
$eval = true
#$evalFilename = "#$deepracerRepo/robomaker-eval.log"
$evalFilename = "#$deepracerRepo/train.log"

tracks = [
# "easy_track"
#              "medium_track",
#             "hard_loopy_track",
#             "hard_speed_track",
#             "hard_track",
            #"AWS_track",
            #"Bowtie_track",
            #"China_track",
#             "Mexico_track",
            "Canada_Training",
#            "China_eval_track",
#            "H_track",
#            "London_Loop_Train",
#            "London_Loop_track",
#            "New_York_Eval_Track",
            #"New_York_Track",
            #"Oval_track",
            #"Straight_track",
            #"Tokyo_Training_track",
            #"Virtual_May19_Train_track",
            #"reinvent_base",
          ]

def executeCmd(cmd, process = false)
    if process
      puts "Execute process #{cmd}"
      pid = spawn(cmd)
      Process.detach(pid)
    else
      puts "Execute command #{cmd}"
      system(cmd)
    end
end

def executeSed(cmd)
    if $isMac
	    cmd = "sed -i '.bak' " + cmd
    else
	    cmd = "sed -i " + cmd
    end
    executeCmd(cmd)
end

def configureSagemaker(track)
  puts "Configure Sagemaker for #{track}"
  filepath = "#$deepracerRepo/rl_coach/env.sh"
  executeSed("'s/WORLD_NAME.*/WORLD_NAME=#{track}/g' #{filepath}")
  executeSed("'s;S3_ENDPOINT_URL=.*;S3_ENDPOINT_URL=#$minioIP;g' #{filepath}")
  if $isMac
      executeCmd("sed -i '.bak' 's/\(readlink/\(greadlink/g' #{filepath}")
  end
  filepath = "#$deepracerRepo/rl_coach/rl_deepracer_coach_robomaker.py"
  executeSed("'s/instance_type =.*/instance_type = \"#$instanceType\"/g' #{filepath}")
  executeSed("'s;image_name=.*;image_name=\"#$sagemakerImage\",;g' #{filepath}")
  filepath = "#$deepracerRepo/robomaker.env"
  executeSed("'s;S3_ENDPOINT_URL=.*;S3_ENDPOINT_URL=#$minioIP;g' #{filepath}")
end

def configureRobomaker(track)
  puts "Configure Robomaker for #{track}"
  filepath = "#$deepracerRepo/robomaker.env"
  executeSed("'s/WORLD_NAME.*/WORLD_NAME=#{track}/g' #{filepath}")
end

def startSagemaker(track)
  puts "Start Sagemaker for #{track}"
  stopSagemaker()
  configureSagemaker(track)
  sagemakerHome = "#$deepracerRepo/rl_coach"
  executeCmd("cd #{sagemakerHome} && #$source ./env.sh && nohup #$pythonExe rl_deepracer_coach_robomaker.py >"\
  " sagemaker.log &", true)
  sleep(30)
  executeCmd("docker exec -t $(docker ps | grep sagemaker | cut -d' ' -f1) redis-cli config set client-output-buffer-limit 'slave 5368709120 5368709120 0'")
  executeCmd("docker exec -t $(docker ps | grep sagemaker | cut -d' ' -f1) redis-cli config set maxmemory 5368709120")
end

def stopSagemaker()
  puts "Stop Sagemaker"
  executeCmd("docker logs $(docker ps -q --filter ancestor='#$sagemakerImage') > sagemaker.log")
  executeCmd("docker stop $(docker ps -q --filter ancestor='#$sagemakerImage')")
  pruneContainers()
end

def checkOOM(track)
  sagemakerHome = "#$deepracerRepo/rl_coach"
  filename = "#{sagemakerHome}/sagemaker.log"
  if File.readlines(filename).grep(/OOM/).any?
    message("Sagemaker killed for #{track} training:")
    executeCmd("rm -rf #$modelPath/#$modelS3Prefix/.finished /mnt/data/minio/.minio.sys/buckets/bucket/rl-deepracer-sagemaker/model/.finished")
    stopSagemaker()
    return true
  end
  filename = "#$deepracerRepo/train.log"
  if File.readlines(filename).grep(/Goodbye/).any?
    message("Sagemaker went away for #{track} training:")
    executeCmd("rm -rf #$modelPath/#$modelS3Prefix/.finished /mnt/data/minio/.minio.sys/buckets/bucket/rl-deepracer-sagemaker/model/.finished")
    stopSagemaker()
    return true
  end
  return false
end

def deleteRobomakerContainers()
  executeCmd("sudo rm -rf #$userHome/robo/container/*")
end

def startRobomaker(track)
  puts "Start Robomaker for #{track}"
  stopRobomaker()
  configureRobomaker(track)
  executeCmd("cd #$deepracerRepo && nohup docker run -v"\
  " /home/#$userHome/deepracer/simulation/aws-robomaker-sample-application-deepracer/simulation_ws/src:"\
  "/app/robomaker-deepracer/simulation_ws/src --rm --name dr --env-file ./robomaker.env"\
  " --network sagemaker-local -p 8080:5900 -i #$robomakerImage > robomaker.log &", true)
end

def stopRobomaker()
  puts "Stop Robomaker"
  executeCmd("docker logs $(docker ps -q --filter ancestor='#$robomakerImage') > robomaker.log")
  executeCmd("docker stop $(docker ps -q --filter ancestor='#$robomakerImage')")
  pruneContainers()
end

def backupLogFiles(track, version)
  puts "Backup training log files"
  executeCmd("cd #$deepracerRepo && mv robomaker.log"\
  " #$dataPath/logs/robomaker-train-#$driver-#{track}-#{version}.log")
  currentDT = Time.new
  executeCmd("cd #$deepracerRepo && cp train.log"\
    " #$dataPath/logs/train-#$driver-#{track}-#{version}.log && truncate -s 0 train.log")
  executeCmd("cd #$deepracerRepo && mv rl_coach/sagemaker.log"\
  " #$dataPath/logs/sagemaker-train-#$driver-#{track}-#{version}.log")
end

def startEval(track, eval_track, version)
  puts "Start Robomaker evaluation for #{track}"
  stopRobomaker()
  sagemakerHome = "#$deepracerRepo/rl_coach"
  executeCmd("cd #{sagemakerHome} && #$source ./env.sh && cd #$deepracerRepo && nohup docker run -v"\
  " /home/#$userHome/deepracer/simulation/aws-robomaker-sample-application-deepracer/simulation_ws/src:"\
  "/app/robomaker-deepracer/simulation_ws/src --rm --name dr_e --env-file ./robomaker.env"\
  " --network sagemaker-local -p 8181:5900 -e 'WORLD_NAME=#{eval_track}' -e 'NUMBER_OF_TRIALS=#$evalCount'"\
  " -e 'METRICS_S3_OBJECT_KEY=custom_files/eval_metric.json'"\
  " -e 'MODEL_S3_PREFIX=rl-deepracer-pretrained-#$driver-#{track}-#{version}'"\
  " -i crr0004/deepracer_robomaker:console"\
  " './run.sh build evaluation.launch' > robomaker.log &", true)
end

def backupEvalFiles(track, eval_track, version)
  puts "Backup evaluation log files"
  executeCmd("cd #$deepracerRepo && mv robomaker.log"\
  " #$dataPath/logs/robomaker-eval-#$driver-#{track}-#{eval_track}-#{version}.log")
  executeCmd("mv #$modelPath/custom_files/eval_metric.json "\
  " #$dataPath/logs/eval_metric-#$driver-#{track}-#{eval_track}-#{version}.json")
  executeCmd("cd #$deepracerRepo && cp train.log"\
    " #$dataPath/logs/train-eval-#$driver-#{track}-#{eval_track}-#{version}.log && truncate -s 0 train.log")
end

def pruneContainers()
  puts "Remove stopped containers"
  executeCmd("docker container prune -f")
end

def snapshotAndArchiveModel(track, version)
  executeCmd("cd #$deepracerRepo && #$source ./aws.sh && python dr_util.py -a snapshot --ssuffix #$driver-#{track}-#{version}")
  executeCmd("cd #$deepracerRepo && #$source ./aws.sh && python dr_util.py -a archive --asuffix #$driver-#{track}-#{version}-archive")
  #executeCmd("cd #$deepracerRepo && #$source ./aws.sh && aws s3 sync #$modelPath #$s3Backup")
  executeCmd("cd #$deepracerRepo && #$source ./aws.sh && aws s3 cp #$modelPath/rl-deepracer-pretrained-#$driver-#{track}-#{version} #$s3Backup/rl-deepracer-pretrained-#$driver-#{track}-#{version} --recursive")
  executeCmd("rm -rf #$modelPath/rl-deepracer-pretrained-#$driver-#{track}-#{version}-archive")
end

def pretrainSagemaker(track, version)
  filepath = "#$deepracerRepo/rl_coach/rl_deepracer_coach_robomaker.py"
  executeSed("'s/#\"pretrained_s3_bucket/\"pretrained_s3_bucket/g' #{filepath}")
  executeSed("'s/.*pretrained_s3_prefix.*/\"pretrained_s3_prefix\": \"rl-deepracer-pretrained-#$driver-#{track}-#{version}\",/g' #{filepath}")
end

def message(message)
  currentDT = Time.new
  puts "#{message} " + currentDT.strftime('%F %T')
end

for version in $offset..$versions
  for track in tracks do
    message("Track #{track} training started version #{version}:")
    if $train
      startSagemaker(track)
      sleep($waitTime)
      startRobomaker(track)
      count = $modelCount
      filename = "#$modelPath/#$modelS3Prefix/model/model_#{count}.pb"
      puts "Check if model training has completed #{filename}."
      while !File.file?(filename) do
        puts "Wait #$fileTime seconds for training."
        sleep($fileTime)
#         executeCmd("docker logs $(docker ps -q --filter ancestor='#$robomakerImage') >& ~/aws-deepracer-workshops/log-analysis/logs/robomaker.log")
        killed = checkOOM(track)
        if killed
          break
        end
      end
      message("Track #{track} training completed version #{version}:")
      stopRobomaker()
      stopSagemaker()
      sleep($waitTime)
      backupLogFiles(track, version)
      snapshotAndArchiveModel(track, version)
      pretrainSagemaker(track, version)
      deleteRobomakerContainers()
    end
    if $eval
      for eval_track in tracks do
        message("Evaluating #{eval_track} started:")
        startEval(track, eval_track, version)
        sleep($waitTime)
        puts "Check if model eval has completed."
        timeSoFar = 0
        while !File.foreach($evalFilename).grep(/has died/).any? do
          puts "Wait #$fileTime seconds for evaluation."
          sleep($fileTime)
          timeSoFar += $fileTime
          if timeSoFar > ($evalCount * 90)
            message("Evalution failed on track #{eval_track} for version #{version}.")
            break
          end
        end
        message("Evaluating #{eval_track} completed:")
        stopRobomaker()
        sleep($waitTime)
        backupEvalFiles(track, eval_track, version)
      end
    end
  end
end
