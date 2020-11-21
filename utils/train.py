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
from utils.metrics import DP
from utils.loss import FLPLoss, KLDLoss
from models.simple_vae import VAE, VAE_Fair

parser = argparse.ArgumentParser(description='vae.pytorch')
parser.add_argument('--logdir', type=str, default="./log/vae-123")
parser.add_argument('--batch_train', type=int, default=64)
parser.add_argument('--batch_test', type=int, default=16)
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--gpu', type=str, default="0")
parser.add_argument('--initial_lr', type=float, default=0.0005)
parser.add_argument('--alpha', type=float, default=1.0)
parser.add_argument('--beta', type=float, default=0.5)
parser.add_argument('--gamma', type=float, default=1.0)
parser.add_argument('--model', type=str, default="vae-123", choices=["vae-123", "vae-345", "pvae"])
parser.add_argument('--fair', action='store_true')
args = parser.parse_args()

# Set GPU (Single GPU usage is only supported so far)
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


def compute_kernel(x, y):
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)
    x = x.unsqueeze(1) # (x_size, 1, dim)
    y = y.unsqueeze(0) # (1, y_size, dim)
    tiled_x = x.expand(x_size, y_size, dim)
    tiled_y = y.expand(x_size, y_size, dim)
    kernel_input = (tiled_x - tiled_y).pow(2).mean(2)/float(dim)
    return torch.exp(-kernel_input) # (x_size, y_size)

def compute_mmd(x, y):
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    mmd = x_kernel.mean() + y_kernel.mean() - 2*xy_kernel.mean()
    return mmd

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

# Reconstruction loss
if args.model == "pvae":
    reconst_criterion = nn.MSELoss(reduction='sum')
elif args.model == "vae-123" or args.model == "vae-345":
    reconst_criterion = FLPLoss(args.model, device, reduction='sum')
# KLD loss
kld_criterion = KLDLoss(reduction='sum')

true_samples_normal = torch.distributions.normal.Normal(torch.zeros(args.batch_train, 100), torch.ones(args.batch_train,100))
true_samples_bernoulli = torch.distributions.bernoulli.Bernoulli(probs = torch.tensor(0.5).repeat(args.batch_train))

# Binary Cross Entropy Loss for predicting sensitive attribute
if args.fair:
    sensitive_attr_criterion = nn.BCELoss() 

# Solver
optimizer = optim.Adam(model.parameters(), lr=args.initial_lr)
# Scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.5, last_epoch=-1)
# Log
logdir = args.logdir
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
            model.train(True)
            logger.write(f"\n----- Epoch {epoch+1} -----")
        else:
            model.train(False)

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
                t = t[:, 20].to(device, dtype=torch.float)  # in the attribute list, 'Male' attribute is located at index 20

                if args.fair:
                    rec_x, mean, logvar, latent_z, sens_attr_pred = model(x)
                else:
                    rec_x, mean, logvar, latent_z = model(x)
                
                # Calc loss
                reconst_loss = reconst_criterion(x, rec_x)
                kld_loss = kld_criterion(mean, logvar)

                true_samples = torch.cat((true_samples_normal.sample(), true_samples_bernoulli.sample().unsqueeze(dim=1)) , dim=1).to(device).detach()
                if not args.fair:
                    true_samples = true_samples_normal.sample().to(device).detach()

                mmd_loss = compute_mmd(torch.cat((latent_z, sens_attr_pred), dim=1), true_samples)  if args.fair else compute_mmd(latent_z, true_samples) 
                sens_attr_loss = sensitive_attr_criterion(sens_attr_pred.squeeze(), t) if args.fair else 0.0

                # loss = args.alpha * kld_loss + args.beta * reconst_loss + args.gamma * sens_attr_loss + mmd_loss
                loss = 2000 * mmd_loss + 0.005 * reconst_loss +  sens_attr_loss

                loss.backward()
                optimizer.step()

                # Visualize
                if i == 0 and x.size(0) >= 64:
                    imsave(x, rec_x, os.path.join(logdir, f"epoch{epoch+1}", f"train.png"), 8, 8)
    
            elif phase == "test":
                with torch.no_grad():
                    optimizer.zero_grad()

                    # Pass forward
                    x = x.to(device)
                    t = t[:, 20].to(device, dtype=torch.float)  # in the attribute list, 'Male' attribute is located at index 20

                    if args.fair:
                        rec_x, mean, logvar, latent_z, sens_attr_pred = model(x)
                    else:
                        rec_x, mean, logvar, latent_z = model(x)    
                    

                    # Calc loss
                    reconst_loss = reconst_criterion(x, rec_x)
                    kld_loss = kld_criterion(mean, logvar)

                    true_samples = torch.cat((true_samples_normal.sample(), true_samples_bernoulli.sample().unsqueeze(dim=1)) , dim=1).to(device).detach()
                    mmd_loss = compute_mmd(torch.cat((latent_z, sens_attr_pred), dim=1), true_samples)  if args.fair else 0.0 

                    sens_attr_loss = sensitive_attr_criterion(sens_attr_pred.squeeze(), t) if args.fair else 0.0
                    
                    # loss = args.alpha * kld_loss + args.beta * reconst_loss + args.gamma * sens_attr_loss
                    loss = 2000 * mmd_loss + 0.005 * reconst_loss +  sens_attr_loss

                    # Visualize
                    if x.size(0) >= 16:
                        imsave(x, rec_x, os.path.join(logdir, f"epoch{epoch+1}", f"test-{i}.png"), 4, 4)

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
torch.save(model.state_dict(),\
    os.path.join(logdir, 'final_model.pth'))


