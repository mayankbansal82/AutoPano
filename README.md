# AutoPano

## Overview
This project explores the creation of seamless panoramic images from multiple overlapping images using both traditional computer vision techniques and deep learning methods. The goal is to stitch images together with minimal artifacts, achieving a wide-angle view.

## Data Collection
Images were captured in sets with approximately 30-50% overlap. Each set contained at least 3 images, resized and formatted in JPG, stored in custom directories within the training data folder.

## Traditional Approach
- **Corner Detection**: Utilized Harris or Shi-Tomasi corner detection methods.
- **ANMS**: Implemented Adaptive Non-Maximal Suppression for even corner distribution.
- **Feature Descriptor and Matching**: Described keypoints with a feature vector and matched features between images.
- **RANSAC**: Applied RANSAC for outlier rejection and robust homography estimation.
- **Blending**: Explored seamless blending techniques for smooth panorama stitching.

## Deep Learning Approach
- **Data Generation**: Generated synthetic pairs of images with known homography from the MSCOCO dataset for training HomographyNet.
- **Supervised Learning**: Designed a CNN model to estimate homography directly, trained with an L2 loss function.
- **Unsupervised Learning**: Implemented an unsupervised method predicting homography for warping without using labels, focusing on photometric loss.

## Results
The project demonstrates the effectiveness of both approaches in creating high-quality panoramas, with deep learning methods providing robust and generalized solutions.

### Traditional Approach
![Before Calibration](https://github.com/mayankbansal82/AutoPano/blob/main/Phase1/Images/1.jpg))
![Before Calibration](https://github.com/mayankbansal82/AutoPano/blob/main/Phase1/Images/2.jpg)
![Before Calibration](https://github.com/mayankbansal82/AutoPano/blob/main/Phase1/Images/3.jpg)
![Before Calibration](https://github.com/mayankbansal82/AutoPano/blob/main/Phase1/Images/mypano.png)

The above image show the results of stitching the first three input images.

### Deep Learning Approach
#### Supervised Approach
![Before Calibration](https://github.com/mayankbansal82/AutoPano/blob/main/Phase2/Images/Train_loss.png)
![Before Calibration](https://github.com/mayankbansal82/AutoPano/blob/main/Phase2/Images/Val_loss.png)
![Before Calibration](https://github.com/mayankbansal82/AutoPano/blob/main/Phase2/Images/Result1.png)
![Before Calibration](https://github.com/mayankbansal82/AutoPano/blob/main/Phase2/Images/Result2.png)

In the last two images, the third part of each image shows the result of calculating homography between the first two parts using supervised approach and then stitching the images together. 

