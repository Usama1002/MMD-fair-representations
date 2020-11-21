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
from models.simple_vae import VAE, VAE_Fair, MLP
from sklearn.metrics import accuracy_score
from utils.metrics import *

parser = argparse.ArgumentParser(description='vae.pytorch')
parser.add_argument('--logdir', type=str, default="./log/vae-123")
parser.add_argument('--batch_train', type=int, default=64)
parser.add_argument('--batch_test', type=int, default=1)
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--gpu', type=str, default="0")
parser.add_argument('--initial_lr', type=float, default=0.0005)
parser.add_argument('--model', type=str, default="vae-123", choices=["vae-123", "vae-345", "pvae"])
parser.add_argument('--fair', action='store_true')
args = parser.parse_args()

# Set GPU (Single GPU usage is only supported so far)
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# -----------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------

# TODO: no gradient from decoder to 'b'

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Dataloader
dataloaders = get_celeba_loaders(args.batch_train, args.batch_test)
# Model

if args.fair:
    model = VAE_Fair(device=device).to(device)
    print("Using fair model")
else:
    model = VAE(device=device).to(device)
    print("Using unfair model")

MLP_Classifier = MLP(100).to(device)


MLP_Classifier.load_state_dict(torch.load(os.path.join(args.logdir, 'classifier/MLP_Classifier_model.pth')))

MLP_Classifier.train(False)
MLP_Classifier.eval()

model.load_state_dict(torch.load(os.path.join(args.logdir, 'final_model.pth')))

model.train(False)
model.eval()

# criterion used is BCE
y_criterion = nn.BCELoss()


# Log
logdir = args.logdir + "/classifier/"
if not os.path.exists(logdir):
    os.makedirs(logdir)
# Logger
logger = Logger(os.path.join(logdir, "log.txt"))
# History
history = {"train": [], "test": []}

# Save config
logger.write('----- Options ------')
for k, v in sorted(vars(args).items()):
    logger.write('%s: %s' % (str(k), str(v)))

sensitive_attribute = []
y = []
y_pred = []
number = 0
accuracy = []

for i, (x,target) in enumerate(dataloaders["test"]):
    # print(target[:,20])
    # Pass forward
    x = x.to(device)
    t = target[:, 18].to(device, dtype=torch.float) #if (args.batch_test < 1) else target[18].to(device, dtype=torch.float)  # in the attribute list, 'Heavy_Makeup' attribute is located at index 18

    a = target[:, 20].to(device, dtype=torch.float) #if (args.batch_test < 1) else target[20].to(device, dtype=torch.float)

    latent_z = model.encode(x)
    
    y_predicted = MLP_Classifier(latent_z)
    number = number + 1

    # print(y_predicted.squeeze().cpu().data.numpy())

    sensitive_attribute.append(a.cpu().data.numpy()[0])
    y.append(t.cpu().data.numpy()[0])
    y_pred.append(y_predicted.squeeze().cpu().data.numpy())
    # accuracy.append(accuracy_score(torch.round(y_predicted.squeeze()).cpu().data.numpy() , t.cpu().data.numpy()))
    # accuracy.append(accuracy_score(torch.round(y_predicted.squeeze()).cpu().data.numpy() , t.cpu().data.numpy()))
    # print("accuracy: " , accuracy_score(torch.round(y_predicted.squeeze()).cpu().data.numpy() , t.cpu().data.numpy()) )


# print(np.mean(accuracy))
print(Equality_of_Opportunity(np.array(y), np.array(y_pred), np.array(sensitive_attribute)))
print(DP(np.array(y_pred), np.array(sensitive_attribute)))

