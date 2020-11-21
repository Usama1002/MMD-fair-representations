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

parser = argparse.ArgumentParser(description='vae.pytorch')
parser.add_argument('--logdir', type=str, default="./log/vae-123")
parser.add_argument('--batch_train', type=int, default=64)
parser.add_argument('--batch_test', type=int, default=16)
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

model.load_state_dict(torch.load(os.path.join(args.logdir, 'final_model.pth')))

model.train(False)

for p in model.parameters():
    p.requires_grad = False

# criterion used is BCE
y_criterion = nn.BCELoss()

# Solver
optimizer = optim.Adam(MLP_Classifier.parameters(), lr=args.initial_lr)
# Scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.5, last_epoch=-1)
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

# Start training
for epoch in range(args.epochs):
    for phase in ["train", "test"]:
        if phase == "train":
            MLP_Classifier.train(True)
            logger.write(f"\n----- Epoch {epoch+1} -----")
        else:
            MLP_Classifier.train(False)

        # Loss
        running_loss = 0.0
        # Data num
        data_num = 0

        # Train
        for i, (x,t) in enumerate(dataloaders[phase]):
            # Optimize params
            if phase == "train":
                optimizer.zero_grad()

                # Pass forward
                x = x.to(device)
                t = t[:, 18].to(device, dtype=torch.float)  # in the attribute list, 'Heavy_Makeup' attribute is located at index 18

                latent_z = model.encode(x)


                y_predicted = MLP_Classifier(latent_z)

                loss = y_criterion(y_predicted.squeeze(), t)

                loss.backward()
                optimizer.step()

            elif phase == "test":
                with torch.no_grad():
                    optimizer.zero_grad()

                    # Pass forward
                    x = x.to(device)
                    t = t[:, 18].to(device, dtype=torch.float)  # in the attribute list, 'Heavy_Makeup' attribute is located at index 18

                    latent_z = model.encode(x)
                    
                    y_predicted = MLP_Classifier(latent_z)

                    loss = y_criterion(y_predicted.squeeze(), t)

            # Add stats
            running_loss += loss # * x.size(0)
            data_num += x.size(0)

        # Log
        epoch_loss = running_loss / data_num
        logger.write(f"{phase} Loss : {epoch_loss:.4f}")
        history[phase].append(epoch_loss)

        if phase == "test":
            plot_loss(logdir, history)

# Save the model
torch.save(MLP_Classifier.state_dict(),\
    os.path.join(logdir, 'MLP_Classifier_model.pth'))


