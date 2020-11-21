# 'vae-123', 'vae-345', 'pvae'
model="pvae"
logdir="./log_unfair_mmd/gamma_500_new/${model}"
gpu="0"

python3.6 -u utils/train_classifier.py --model $model --logdir $logdir --gpu $gpu

exit 0
