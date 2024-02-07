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
#import kornia  # You can use this to get the transform and warp in this project
from kornia.geometry.transform import HomographyWarper as HW
# Don't generate pyc codes
sys.dont_write_bytecode = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def lossFn(P_B_dash,P_B):
    loss=torch.mean(torch.abs(P_B_dash-P_B))
    return loss

def get_wrapped_images(P_A, H_batch):
    P_B_dash=HW(128,128)(P_A,H_batch)
    print(P_B_dash)
    return P_B_dash



def tensorDLT(C_A,H4pt):
    batch_size=H4pt.shape[0]
    C_B=C_A+H4pt.view(batch_size,4,2)
    H_batch=[]
    for i in range(batch_size):
        c_a=C_A[i,:,:]
        c_b=C_B[i,:,:]
        A=[]
        b=[]
        for j in range(4):
            a = [ [0, 0, 0, -c_a[j, 0], -c_a[j, 1], -1, c_b[j, 1]*c_a[j, 0], c_b[j, 1]*c_a[j, 1]], 
                    [c_a[j, 0], c_a[j, 1], 1, 0, 0, 0, -c_b[j, 0]*c_a[j, 0], -c_b[j, 0]*c_a[j, 1]] ]
            bs = [[-c_b[j, 1]], [c_b[j, 0]]]
            A.append(a)
            b.append(bs)
        A=torch.tensor(A,dtype=torch.float32).to(device).reshape(8,8)
        A.requires_grad=True
        b=torch.tensor(b,dtype=torch.float32).to(device).reshape(8,1)
        b.requires_grad=True
        A_pseudo_inv = torch.pinverse(A)
        x = torch.matmul(A_pseudo_inv, b)
        h=torch.cat((x,torch.tensor([[1]],dtype=torch.float32).to(device)),dim=0)
        H_batch.append(h.reshape(3,3))
    H_batch=torch.stack(H_batch,dim=0)

    return H_batch

class HomographyModel(nn.Module):
    def __init__(self):
        super(HomographyModel, self).__init__()
        self.model = Net()
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        P_A,P_B,C_A,I_A,Combined = batch
        C_A.requires_grad=True
        H4pt = self.model(Combined)
        H_batch=tensorDLT(C_A,H4pt)
        P_B_dash=get_wrapped_images(P_A, H_batch)
        loss = lossFn(P_B_dash,P_B)
        return loss

    def validation_step(self, batch):
        P_A,P_B,C_A,I_A,Combined = batch
        H4pt = self.model(Combined)
        H_batch=tensorDLT(C_A,H4pt)
        P_B_dash=get_wrapped_images(P_A, H_batch)
        loss = lossFn(P_B_dash,P_B)
        return {"loss": loss.detach()}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        logs = {"val_loss": avg_loss}
        return {"avg_val_loss": avg_loss, "log": logs}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], loss: {:.4f}, acc: {:.4f}".format(epoch, result['loss'], result['acc']))


class Net(nn.Module):
    def __init__(self):
        """
        Inputs:
        InputSize - Size of the Input
        OutputSize - Size of the Output
        """
        super().__init__()
        #############################
        # Fill your network initialization of choice here!
        #############################
        self.conv1=nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv2=nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,stride=2)
        )
        self.conv3=nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv4=nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,stride=2)
        )
        self.conv5=nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.conv6=nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,stride=2)
        )
        self.conv7=nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.conv8=nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        #droupout with a probability of 0.5
        self.dropout=nn.Dropout(0.4)
        #fully connected layer
        self.fc1=nn.Linear(128*16*16,1024)
        self.fc2=nn.Linear(1024,8)
        


        

    

    def forward(self, x):
        """
        Input:
        xa is a MiniBatch of the image a
        xb is a MiniBatch of the image b
        Outputs:
        out - output of the network
        """
        #############################
        # Fill your network structure of choice here!
        #############################
        
        #pass through the first layer
        x=self.conv1(x)
        x=self.conv2(x)
        x=self.conv3(x)
        x=self.conv4(x)
        x=self.conv5(x)
        x=self.conv6(x)
        x=self.conv7(x)
        x=self.conv8(x)
        #dropout
        x=self.dropout(x)
        #flatten the output
        x=x.view(x.size(0),-1)
        #fully connected layer
        x=self.fc1(x)
        x=self.dropout(x)
        x=self.fc2(x)
        #output
        out=x
        return out


# #test the network
# model=Net()
# model.to(device)

# #create a random input
# x=torch.randn(4,2,128,128)
# x=x.to(device)
# out=model(x)

# print(out)

# C_A=torch.randn(4,4,2)

# C_A=C_A.to(device)

# H_batch=tensorDLT(C_A,out)

# print(H_batch)

# P_A=torch.randn(4,1,128,128)
# P_A=P_A.to(device)

# P_B=torch.randn(4,1,128,128)
# P_B=P_B.to(device)

# P_B_dash,P_B=get_wrapped_images(P_A, P_B, H_batch)

# print(P_B_dash.shape)

# loss = lossFn(P_B_dash,P_B)
# print(loss)