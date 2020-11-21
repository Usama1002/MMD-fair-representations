import os
import sys

sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

sys.path.append(".")

from utils.data import get_celeba_loaders
from utils.vis import plot_loss, imsave, Logger
from utils.loss import FLPLoss, KLDLoss
from models.simple_vae import VAE, VAE_Fair

from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='vae.pytorch')
parser.add_argument('--logdir', type=str, default="./log/vae-123")
parser.add_argument('--batch_train', type=int, default=64)
parser.add_argument('--batch_test', type=int, default=200)
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--gpu', type=str, default="0")
parser.add_argument('--initial_lr', type=float, default=0.0005)
parser.add_argument('--alpha', type=float, default=1.0)
parser.add_argument('--beta', type=float, default=0.5)
parser.add_argument('--model', type=str, default="vae-123", choices=["vae-123", "vae-345", "pvae"])
parser.add_argument('--fair', action='store_true')
args = parser.parse_args()

# Set GPU (Single GPU usage is only supported so far)
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Dataloader
dataloaders = get_celeba_loaders(args.batch_train, args.batch_test)
# Model

logdir = args.logdir 

if args.fair:
    model = VAE_Fair(device=device).to(device)
    print("Using fair model")
else:
    model = VAE(device=device).to(device)
    print("Using unfair model")


model.load_state_dict(torch.load(os.path.join(logdir, 'final_model.pth')))

model.eval()


# Reconstruction loss
# if args.model == "pvae":
#     reconst_criterion = nn.MSELoss(reduction='sum')
# elif args.model == "vae-123" or args.model == "vae-345":
#     reconst_criterion = FLPLoss(args.model, device, reduction='sum')
# KLD loss
# kld_criterion = KLDLoss(reduction='sum')

# Binary Cross Entropy Loss for predicting sensitive attribute
# if args.fair:
#     sensitive_attr_criterion = nn.BCELoss() 

# Solver
# optimizer = optim.Adam(model.parameters(), lr=args.initial_lr)
# Scheduler
# scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.5, last_epoch=-1)
# Log
# logdir = args.logdir
# if not os.path.exists(logdir):
#     os.makedirs(logdir)
# Logger
# logger = Logger(os.path.join(logdir, "log.txt"))
# History
history = {"train": [], "test": []}

# Save config
# logger.write('----- Options ------')
# for k, v in sorted(vars(args).items()):
#     logger.write('%s: %s' % (str(k), str(v)))

# Start training
# model.train(False)

import cv2

for i, (x,t) in enumerate(dataloaders["test"]):
    x = x.to(device)
    t = t[:, 20].to(device, dtype=torch.float)  # in the attribute list, 'Male' attribute is located at index 20
    if args.fair:
        rec_x, mean, logvar,latent_z, sens_attr_pred = model(x)
    else:
        rec_x, mean, logvar, latent_z = model(x)

    print("accuracy: " , accuracy_score(torch.round(sens_attr_pred.squeeze()).cpu().data.numpy() , t.cpu().data.numpy()) )
