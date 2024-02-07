#!/usr/bin/env python3

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from Network.Network import HomographyModel, loss_fn, HNet
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
from PIL import Image
from tqdm import tqdm

# Don't generate pyc codes
sys.dont_write_bytecode = True


class HomographyDataset(Dataset):
    def __init__(self, img_dir_patch_a, img_dir_patch_b, labels_file, transform=None):
        self.img_dir_patch_a = img_dir_patch_a
        self.img_dir_patch_b = img_dir_patch_b
        self.labels = np.load(labels_file)
        self.transform = transform
        self.img_filenames = [f for f in os.listdir(img_dir_patch_a) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.img_filenames)

    def __getitem__(self, idx):
        img_name_a = os.path.join(self.img_dir_patch_a, self.img_filenames[idx])
        img_name_b = os.path.join(self.img_dir_patch_b, self.img_filenames[idx])
        
        # Read Data
        imgA = cv2.imread(img_name_a, cv2.IMREAD_GRAYSCALE)
        imgB = cv2.imread(img_name_b, cv2.IMREAD_GRAYSCALE)

        # Normalize Data and convert to torch tensors
        imgA = torch.from_numpy((imgA.astype(np.float32) - 127.5) / 127.5)
        imgB = torch.from_numpy((imgB.astype(np.float32) - 127.5) / 127.5)

        # if self.transform:
        #     imgA = self.transform(imgA)
        #     imgB = self.transform(imgB)

         # Stack grayscale images
        image = torch.stack((imgA, imgB), dim=0)


        label = torch.from_numpy(self.labels[idx].astype(np.float32)/32.0)

        # image = torch.cat((image_a, image_b), dim=0)
        return image, label



def PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile):
    """
    Prints all stats with all arguments
    """
    print('Number of Epochs Training will run for ' + str(NumEpochs))
    print('Factor of reduction in training data is ' + str(DivTrain))
    print('Mini Batch Size ' + str(MiniBatchSize))
    print('Number of Training Images ' + str(NumTrainSamples))
    if LatestFile is not None:
        print('Loading latest checkpoint with the name ' + LatestFile) 


def TrainOperation(train_dataloader, val_dataloader, NumEpochs, CheckPointPath,LatestFile, LogsPath):
    model = HNet()
    Optimizer = Adam(model.parameters(), lr=1e-3)

    # Tensorboard
    # Create a summary to monitor loss tensor
    Writer = SummaryWriter(LogsPath)

    if LatestFile is not None:
        CheckPoint = torch.load(CheckPointPath + LatestFile + '.ckpt')
        # Extract only numbers from the name
        StartEpoch = int(''.join(c for c in LatestFile.split('a')[0] if c.isdigit()))
        model.load_state_dict(CheckPoint['model_state_dict'])
        print('Loaded latest checkpoint with the name ' + LatestFile + '....')
    else:
        StartEpoch = 0
        print('New model initialized....')

    for Epochs in tqdm(range(StartEpoch, NumEpochs)):
        model.train()
        epoch_loss_train = 0
        for images,labels in train_dataloader:
            labels = labels.reshape(labels.shape[0], -1)

            # Predict output with forward pass
            LossThisBatch = model.training_step((images,labels))

            Optimizer.zero_grad()
            LossThisBatch.backward()
            Optimizer.step()

            result = model.validation_step((images,labels))
            epoch_loss_train += result["loss"].item()

        avg_loss_train = epoch_loss_train / len(train_dataloader)
        Writer.add_scalar('LossEveryEpoch/Train',  avg_loss_train, Epochs)


        model.eval() 
        epoch_loss_val = 0
        with torch.no_grad():
            for images, labels in val_dataloader:
                labels = labels.reshape(labels.shape[0], -1)
                result = model.validation_step((images, labels))
                epoch_loss_val += result["loss"].item()

        avg_loss_val = epoch_loss_val / len(val_dataloader)
        Writer.add_scalar('LossEveryEpoch/Val', avg_loss_val, Epochs)

        # If you don't flush the tensorboard doesn't update until a lot of iterations!
        Writer.flush()

        # Save the model every 10 epochs
        if Epochs % 10 == 0 or Epochs == NumEpochs - 1:
            SaveName = CheckPointPath + str(Epochs) + 'model.ckpt'
            torch.save({'epoch': Epochs, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': Optimizer.state_dict(), 'loss': LossThisBatch}, SaveName)
            print('\n' + SaveName + ' Model Saved...')

    Writer.close()



def main():
    """
    Inputs: 
    None
    Outputs:
    Runs the Training and testing code based on the Flag
    """
    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--CheckPointPath', default='../Checkpoints/', help='Path to save Checkpoints, Default: ../Checkpoints/ResNeXt')
    Parser.add_argument('--NumEpochs', type=int, default=50, help='Number of Epochs to Train for, Default:50')
    Parser.add_argument('--DivTrain', type=int, default=1, help='Factor to reduce Train data by per epoch, Default:1')
    Parser.add_argument('--MiniBatchSize', type=int, default=64, help='Size of the MiniBatch to use, Default:128')
    Parser.add_argument('--LoadCheckPoint', type=int, default=0, help='Load Model from latest Checkpoint from CheckPointsPath?, Default:0')
    Parser.add_argument('--LogsPath', default='Logs/', help='Path to save Logs for Tensorboard, Default=Logs/modified')
    # TrainSet = torchvision.datasets.CIFAR10(root='./data', train=True,
    #                                     download=True, transform=ToTensor())

    Args = Parser.parse_args()
    NumEpochs = Args.NumEpochs
    DivTrain = float(Args.DivTrain)
    MiniBatchSize = Args.MiniBatchSize
    LoadCheckPoint = Args.LoadCheckPoint
    CheckPointPath = Args.CheckPointPath
    LogsPath = Args.LogsPath
    TestLabelsPath = "./TxtFiles/LabelsTest.txt"
    
    # Setup all needed parameters including file reading
    SaveCheckPoint, ImageSize, NumTrainSamples, TrainLabels, NumClasses = SetupAll(TestLabelsPath, CheckPointPath)


    # Find Latest Checkpoint File
    if LoadCheckPoint==1:
        LatestFile = FindLatestModel(CheckPointPath)
    else:
        LatestFile = None
    
    # Pretty print stats
    PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile)

    # Transformations, adjust as needed
    transform = transforms.Compose([
    transforms.ToTensor(),
    # Add more transformations if required
    ])

    # Initialize dataset and dataloader
    train_dataset = HomographyDataset(
    img_dir_patch_a='../Data/modified_train/patchA',
    img_dir_patch_b='../Data/modified_train/patchB',
    labels_file='TxtFiles/modified_train_labels.npy',
    transform=transform
    )

    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)


    # Initialize dataset and dataloader
    val_dataset = HomographyDataset(
    img_dir_patch_a='../Data/modified_val/patchA',
    img_dir_patch_b='../Data/modified_val/patchB',
    labels_file='TxtFiles/modified_val_labels.npy',
    transform=transform
    )

    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=True)

    TrainOperation(train_dataloader, val_dataloader, NumEpochs, CheckPointPath,LatestFile, LogsPath)

    
if __name__ == '__main__':
    main()        






