#!/usr/bin/env python

"""
RBE/CS Fall 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Lening Li (lli4@wpi.edu)
Teaching Assistant in Robotics Engineering,
Worcester Polytechnic Institute
"""


# Dependencies:
# opencv, do (pip install opencv-python)
# skimage, do (apt install python-skimage)
# termcolor, do (pip install termcolor)

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import torch.optim as optim
from Network.Network_Supervised import HomographyModel
import cv2
import sys
import os
import numpy as np
import random
import skimage
import PIL
import os
import glob
import random
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
import numpy as np
import time
from Misc.MiscUtils import *
from Misc.DataUtils import *
from torchvision.transforms import ToTensor
import argparse
import shutil
import string
from termcolor import colored, cprint
import math as m
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader



def PrettyPrint(NumEpochs, MiniBatchSize, NumTrainSamples):
    """
    Prints all stats with all arguments
    """
    print("Number of Epochs Training will run for " + str(NumEpochs))
    print("Mini Batch Size " + str(MiniBatchSize))
    print("Number of Training Images " + str(NumTrainSamples))


class HomographyDataset(Dataset):
    def __init__(self, BasePath):
        self.BasePath = BasePath
        self.Imagepath=BasePath+os.sep+"Images"
        self.Labelpath=BasePath+os.sep+"Labels"
        self.label_list=[self.Labelpath+os.sep+label for label in os.listdir(self.Labelpath)]
        self.label_list.sort()

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, idx):
        label_path=self.label_list[idx]
        
        label=np.load(label_path)
        
        label=label.reshape(-1)
        

        label_num=label_path.split(os.sep)[-1].split(".")[0]
        image_A_path=self.Imagepath+os.sep+label_num+"_A.jpg"
        image_B_path=self.Imagepath+os.sep+label_num+"_B.jpg"
        
        image_A=cv2.imread(image_A_path,0)
        image_B=cv2.imread(image_B_path,0)
        

        image=np.stack([image_A,image_B],axis=0)
        image=image.astype(np.float32)
        label=label.astype(np.float32)
        
        image=torch.from_numpy(image)
        label=torch.from_numpy(label)



        return image,label

def TrainOperation(
    NumEpochs,
    dataloader,
    SaveCheckPoint,
    CheckPointPath,
    LogsPath,
    val_dataloader
):
    """
    Inputs:
    ImgPH is the Input Image placeholder
    DirNamesTrain - Variable with Subfolder paths to train files
    TrainCoordinates - Coordinates corresponding to Train/Test
    NumTrainSamples - length(Train)
    ImageSize - Size of the image
    NumEpochs - Number of passes through the Train data
    MiniBatchSize is the size of the MiniBatch
    SaveCheckPoint - Save checkpoint every SaveCheckPoint iteration in every epoch, checkpoint saved automatically after every epoch
    CheckPointPath - Path to save checkpoints/model
    DivTrain - Divide the data by this number for Epoch calculation, use if you have a lot of dataor for debugging code
    LatestFile - Latest checkpointfile to continue training
    BasePath - Path to COCO folder without "/" at the end
    LogsPath - Path to save Tensorboard Logs
        ModelType - Supervised or Unsupervised Model
    Outputs:
    Saves Trained network in CheckPointPath and Logs to LogsPath
    """
    # Predict output with forward pass
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = HomographyModel()
    model.to(device)
    ###############################################
    # Fill your optimizer of choice here!
    ###############################################
    Optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9)

    
    scheduler = optim.lr_scheduler.StepLR(Optimizer, step_size=30000, gamma=0.1)

    # Tensorboard
    # Create a summary to monitor loss tensor
    Writer = SummaryWriter(LogsPath)

   
    print("New model initialized....")

    for Epochs in tqdm(range(NumEpochs)):
        total_loss_epoch = 0
        for i,(images,labels) in enumerate(dataloader):
            
            images=images.to(device)
            labels=labels.to(device)
            # Predict output with forward pass
            model.train()            
            Loss= model.training_step((images,labels))
            

            Optimizer.zero_grad()
            Loss.backward()
            Optimizer.step()
            scheduler.step()

            result=model.validation_step((images,labels))
            total_loss_epoch+=result["loss"].item()
        
        print(total_loss_epoch)

        # Tensorboard
        Writer.add_scalar(
            "LossEveryEpoch/Train",
            total_loss_epoch/len(dataloader),
            Epochs ,
        )
        total_loss_epoch_val=0
        for i,(images,labels) in enumerate(val_dataloader):
            images=images.to(device)
            labels=labels.to(device)
            model.eval()
            result=model.validation_step((images,labels))
            total_loss_epoch_val+=result["loss"].item()
            
        Writer.add_scalar(
            "LossEveryEpoch/Val",
            total_loss_epoch_val/len(val_dataloader),
            Epochs ,
        )
        

        
        # If you don't flush the tensorboard doesn't update until a lot of iterations!
        Writer.flush()

        # Save model every epoch
        if Epochs % 10 == 0 or Epochs == NumEpochs - 1:
            SaveName = CheckPointPath + str(Epochs) + "model.ckpt"
            torch.save(
                {
                    "epoch": Epochs,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": Optimizer.state_dict(),
                    "loss": Loss,
                },
                SaveName,
            )
            print("\n" + SaveName + " Model Saved...")


def main():
    """
    Inputs:
    # None
    # Outputs:
    # Runs the Training and testing code based on the Flag
    #"""
    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument(
        "--BasePath",
        default="./Dataset/Train",
        help="Base path of images and Labels",
    )
    Parser.add_argument(
        "--CheckPointPath",
        default="./Supervised/Checkpoints/",
        help="Path to save Checkpoints, Default: ./Checkpoints/",
    )

    
    Parser.add_argument(
        "--NumEpochs",
        type=int,
        default=100,
        help="Number of Epochs to Train for, Default:50",
    )
    
    Parser.add_argument(
        "--MiniBatchSize",
        type=int,
        default=64,
        help="Size of the MiniBatch to use, Default:1",
    )
    
    Parser.add_argument(
        "--LogsPath",
        default="./Supervised/Logs/",
        help="Path to save Logs for Tensorboard, Default=Logs/",
    )

    #if chckpoints folde is not present then create it
    if not os.path.exists("./Supervised/Checkpoints/"):
        os.makedirs("./Supervised/Checkpoints/")
    
    #if logs folder is not present then create it
    if not os.path.exists("./Supervised/Logs/"):
        os.makedirs("./Supervised/Logs/")
    
   

    Args = Parser.parse_args()
    NumEpochs = Args.NumEpochs
    BasePath = Args.BasePath
    MiniBatchSize = Args.MiniBatchSize
    CheckPointPath = Args.CheckPointPath
    LogsPath = Args.LogsPath

    SaveCheckPoint = 1000

    NumTrainSamples = len(glob.glob(BasePath + os.sep + "Labels" + os.sep + "*.npy"))

    # Pretty print stats
    PrettyPrint(NumEpochs, MiniBatchSize, NumTrainSamples)

    train_dataset=HomographyDataset(BasePath)
    train_dataloader=DataLoader(train_dataset,batch_size=MiniBatchSize,shuffle=True,num_workers=8)

    val_dataset=HomographyDataset("./Dataset/Val")
    val_dataloader=DataLoader(val_dataset,batch_size=MiniBatchSize,shuffle=True,num_workers=8)

    TrainOperation(
        NumEpochs,
        train_dataloader,
        SaveCheckPoint,
        CheckPointPath,
        LogsPath,
        val_dataloader
    )


if __name__ == "__main__":
    main()
