# 'vae-123', 'vae-345', 'pvae'
model="pvae"
logdir="./trained_models/temp/${model}"
gpu="0"

python3.6 -u utils/train.py --model $model --logdir $logdir --gpu $gpu 

exit 0
