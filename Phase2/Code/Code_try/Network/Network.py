"""
RBE/CS Fall 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Lening Li (lli4@wpi.edu)
Teaching Assistant in Robotics Engineering,
Worcester Polytechnic Institute
"""

import torch.nn as nn
import sys
import torch
import numpy as np
import torch.nn.functional as F
# import kornia  # You can use this to get the transform and warp in this project

# Don't generate pyc codes
sys.dont_write_bytecode = True


def loss_fn(output, labels):
    ###############################################
    # Fill your loss function of choice here!
    ###############################################

    ###############################################
    # You can use kornia to get the transform and warp in this project
    # Bonus if you implement it yourself
    ###############################################
    criterion = nn.MSELoss()
    loss = criterion(output, labels)
    return loss


class HomographyModel(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images) # Generate predictions
        loss = loss_fn(out, labels) # Calculate loss
        # return loss
        # logs = {"loss": loss}
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = loss_fn(out, labels)
        return {'loss': loss.detach()}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        return {'val_loss': epoch_loss.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], loss: {:.4f}, acc: {:.4f}".format(
            epoch, result['loss'], result['acc']))


class HNet(HomographyModel):
    def __init__(self):
        super(HNet,self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=128 * 16 * 16, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=8),
            nn.Softmax(dim=1)
        )

    
    def forward(self, x):
        # Pass input through sequential layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Flatten the output for the fully connected layers
        x = x.view(x.size(0), -1) # Flatten the tensor
        x = self.fc(x)
        
        return x
        



# class Net(nn.Module):
#     def __init__(self, InputSize, OutputSize):
#         """
#         Inputs:
#         InputSize - Size of the Input
#         OutputSize - Size of the Output
#         """
#         super().__init__()
#         #############################
#         # Fill your network initialization of choice here!
#         #############################
#         ...
#         #############################
#         # You will need to change the input size and output
#         # size for your Spatial transformer network layer!
#         #############################
#         # Spatial transformer localization-network
#         self.localization = nn.Sequential(
#             nn.Conv2d(1, 8, kernel_size=7),
#             nn.MaxPool2d(2, stride=2),
#             nn.ReLU(True),
#             nn.Conv2d(8, 10, kernel_size=5),
#             nn.MaxPool2d(2, stride=2),
#             nn.ReLU(True),
#         )

#         # Regressor for the 3 * 2 affine matrix
#         self.fc_loc = nn.Sequential(
#             nn.Linear(10 * 3 * 3, 32), nn.ReLU(True), nn.Linear(32, 3 * 2)
#         )

#         # Initialize the weights/bias with identity transformation
#         self.fc_loc[2].weight.data.zero_()
#         self.fc_loc[2].bias.data.copy_(
#             torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)
#         )

#     #############################
#     # You will need to change the input size and output
#     # size for your Spatial transformer network layer!
#     #############################
#     def stn(self, x):
#         "Spatial transformer network forward function"
#         xs = self.localization(x)
#         xs = xs.view(-1, 10 * 3 * 3)
#         theta = self.fc_loc(xs)
#         theta = theta.view(-1, 2, 3)

#         grid = F.affine_grid(theta, x.size())
#         x = F.grid_sample(x, grid)

#         return x

#     def forward(self, xa, xb):
#         """
#         Input:
#         xa is a MiniBatch of the image a
#         xb is a MiniBatch of the image b
#         Outputs:
#         out - output of the network
#         """
#         #############################
#         # Fill your network structure of choice here!
#         #############################
#         return out
