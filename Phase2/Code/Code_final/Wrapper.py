#!/usr/bin/evn python

"""
RBE/CS Fall 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Lening Li (lli4@wpi.edu)
Teaching Assistant in Robotics Engineering,
Worcester Polytechnic Institute
"""


# Code starts here:

import numpy as np
import cv2
from Network.Network_Unsupervised import HomographyModel
import torch
import argparse
import os
import time


# Add any python libraries here

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main(ModelPath, Set, ImageNumber):
    # Add any Command Line arguments here
    # Parser = argparse.ArgumentParser()
    # Parser.add_argument('--NumFeatures', default=100, help='Number of best features to extract from each image, Default:100')

    # Args = Parser.parse_args()
    # NumFeatures = Args.NumFeatures

    """
    Read a set of images for Panorama stitching
    """

    image_A=cv2.imread('./Dataset/'+Set+'/Images/'+str(ImageNumber)+'_A.jpg',0)
    image_B=cv2.imread('./Dataset/'+Set+'/Images/'+str(ImageNumber)+'_B.jpg',0)

    image=np.stack([image_A,image_B],axis=0)
    image=image.astype(np.float32)
    
    
    #image is now 2x128x128 but we need 1x2x128x128
    
    image=torch.from_numpy(image)

    image=image.unsqueeze(0)
    
    image=image.to(device)
    #load the label 1.npy
    label=np.load('./Dataset/'+Set+'/Labels/'+str(ImageNumber)+'.npy')
    """
	Obtain Homography using Deep Learning Model (Supervised and Unsupervised)
	"""
    model=HomographyModel()
    CheckPoint = torch.load(ModelPath)
    model.load_state_dict(CheckPoint['model_state_dict'])
    model.to(device)

    model.eval()
    output=model(image)
    
    output=output.cpu().detach().numpy()
    # print(label.reshape(1,8))
    
    """
	Image Warping + Blending
	Save Panorama output as mypano.png
	"""
    
    corners_A=[[0,0],[128,0],[0,128],[128,128]]

    corners_A=np.array(corners_A)
    print(corners_A)

    corners_B=corners_A+output.reshape(4,2)
    print(corners_B)

    H_BA=(cv2.getPerspectiveTransform(np.float32(corners_A), np.float32(corners_B)))

    final_image_boundaries=np.concatenate((corners_A,corners_B),axis=0)

    [x_min,y_min]=np.min(final_image_boundaries,axis=0)
    [x_max,y_max]=np.max(final_image_boundaries,axis=0)

    x_min=int(x_min)
    y_min=int(y_min)
    x_max=int(x_max)
    y_max=int(y_max)


    translation_matrix=np.array([[1,0,-x_min],[0,1,-y_min],[0,0,1]])

    I_B = cv2.warpPerspective(image_B, np.matmul(translation_matrix,H_BA), (x_max-x_min, y_max-y_min))

    final_image=I_B.copy()
    final_image[-y_min:-y_min+image_A.shape[0],-x_min:-x_min+image_A.shape[1]]=image_A.copy()

    indices=np.where(image_A==[0])

    y=indices[0]-y_min
    x=indices[1]-x_min

    final_image[y,x]=I_B[y,x]

   
    
    C_A=np.load('./Dataset/'+Set+'/Labels/'+str(ImageNumber)+'_C_A.npy')
    corners_A=C_A.reshape(4,2)
    original_image=cv2.imread('./Dataset/'+Set+'/Images/'+str(ImageNumber)+'_I_A.jpg',0)
    corner_B_generated=output.reshape(4,2)+corners_A
    corner_B_actual=label.reshape(4,2)+corners_A

    corner_B_generated=corner_B_generated[[0,1,3,2],:]
    corner_B_actual=corner_B_actual[[0,1,3,2],:]


    for i in range(4):
        cv2.line(original_image,(int(corner_B_generated[i,1]),int(corner_B_generated[i,0])),(int(corner_B_generated[(i+1)%4,1]),int(corner_B_generated[(i+1)%4,0])),(255,0,0),2)
        cv2.line(original_image,(int(corner_B_actual[i,1]),int(corner_B_actual[i,0])),(int(corner_B_actual[(i+1)%4,1]),int(corner_B_actual[(i+1)%4,0])),(0,0,255),2)
    

    
    if not os.path.exists('./Results/'):
        os.makedirs('./Results/') 
    
    num_folders=len(os.listdir('./Results/'))
    
    os.makedirs('./Results/'+str(num_folders+1))
    
    cv2.imwrite('./Results/'+str(num_folders+1)+'/final_image_'+str(num_folders+1)+'.jpg',final_image)
    cv2.imwrite('./Results/'+str(num_folders+1)+'/box_image_'+str(num_folders+1)+'.jpg',original_image)
    cv2.imwrite('./Results/'+str(num_folders+1)+'/image_a_'+str(num_folders+1)+'.jpg',image_A)
    cv2.imwrite('./Results/'+str(num_folders+1)+'/image_b_'+str(num_folders+1)+'.jpg',image_B)


    

if __name__ == "__main__":

    Parser = argparse.ArgumentParser()
    Parser.add_argument('--ModelPath', default='./Supervised/Checkpoints/99model.ckpt', help='Supervised or Unsupervised')
    Parser.add_argument('--Set', default='Train', help='Train or Val or Test')
    Parser.add_argument('--ImageNumber', default=1, help='Image Number to test')
    Args = Parser.parse_args()
    ModelPath = Args.ModelPath
    Set = Args.Set
    ImageNumber = Args.ImageNumber
    main(ModelPath, Set, ImageNumber)


