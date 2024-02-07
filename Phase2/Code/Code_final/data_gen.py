import os
import numpy as np
import cv2
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--Set', default='Train', help='Train or Val or Test')

args = parser.parse_args()
Set = args.Set



#read all the images in  the folder ./Data/Val/ which are .jpg files
images_path = os.listdir('./Data/'+Set+'/')
images_path = [x for x in images_path if x.endswith('.jpg')]
images_path.sort()

image_num=1
number_per_image=1
for i in tqdm(range(number_per_image)):
    for image_path in images_path:
        image = cv2.imread('./Data/'+Set+'/'+image_path)
        image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

        image=cv2.resize(image,(320,240))
        I_A=image.copy()
        
        #get a random patch in the image
        patch_size = 128

        #threshold to  avoid the patch being too close to the border
        threshold = 40

        #get the top left corner of the patch
        x = np.random.randint(threshold, image.shape[0] - patch_size - threshold)
        y = np.random.randint(threshold, image.shape[1] - patch_size - threshold)

        #get the patch
        patch_A = image[x:x+patch_size, y:y+patch_size]

        #corners of the patch
        corners_A = np.array([[x, y], [x+patch_size, y], [x, y+patch_size], [x+patch_size, y+patch_size]])
        

        #get the random perturbation
        max_perturbation = 16
        perturbation = np.random.randint(-max_perturbation, max_perturbation, size=corners_A.shape)
        

        #get the perturbed corners
        corners_B = corners_A + perturbation

        H_BA=np.linalg.inv(cv2.getPerspectiveTransform(np.float32(corners_A), np.float32(corners_B)))

        #get the warped image
        I_B = cv2.warpPerspective(image, H_BA, (image.shape[1], image.shape[0]))

        #get the patch from the warped image
        patch_B = I_B[x:x+patch_size, y:y+patch_size]

        #save the images
        #create a folder called UnSupervised_Dataset/Val 
        if not os.path.exists('./Dataset/'+Set+'/Images/'):
            os.makedirs('./Dataset/'+Set+'/Images/')
        if not os.path.exists('./Dataset/'+Set+'/Labels/'):
            os.makedirs('./Dataset/'+Set+'/Labels/')

        

        cv2.imwrite('./Dataset/'+Set+'/Images/'+str(image_num)+'_A.jpg',patch_A)
        cv2.imwrite('./Dataset/'+Set+'/Images/'+str(image_num)+'_B.jpg',patch_B)
        cv2.imwrite('./Dataset/'+Set+'/Images/'+str(image_num)+'_I_A.jpg',I_A)

        H4pt=corners_B-corners_A
        #save the labels
        np.save('./Dataset/'+Set+'/Labels/'+str(image_num)+'.npy',H4pt)
        np.save('./Dataset/'+Set+'/Labels/'+str(image_num)+'_C_A.npy',corners_A)

        image_num+=1

    






