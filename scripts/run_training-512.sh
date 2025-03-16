timestamp=$(date +"%Y-%m-%d_%H%M%S")
logfile="ldm_training_512_${timestamp}.log"
nohup python main.py --base models/ldm/drsk/config-512-with-vq-f4.yaml -t --gpus 0,1,2,3 --strategy ddp > "$logfile" 2>&1 &