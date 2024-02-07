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
import argparse
import os

# Add any python libraries here

def visualizeAndSaveCorners(image, corners, filename):
    # Make a copy of the image to draw on
    vis_image = image.copy()

    # Draw the corners
    for corner in corners:
        x, y = corner
        cv2.circle(vis_image, (x, y), radius=2, color=(0, 255, 0), thickness=-1)  # Green dot

    # Save the image with corners
    # cv2.imwrite(filename, vis_image)

    # Also display the image if desired
    # cv2.imshow("Corners", vis_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def visualizeAndSaveMatches(image1, keypoints1, image2, keypoints2, matches, filename):
    # Draw the matches
    matched_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    # Save the image with matches
    # cv2.imwrite(filename, matched_image)
    
    # Also display the image if desired
    # cv2.imshow("Matches", matched_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()




def readImages(images_path):
    print(f"Reading images from '{images_path}'")
    images = []
    for file_name in sorted(os.listdir(images_path)):
        image_path = os.path.join(images_path, file_name)
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Error loading {image_path}")
        images.append(image)
    return images

def detectCorners(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray, maxCorners = 10000, qualityLevel = 0.001, minDistance = 15)
    corners = np.int32(corners)

    return corners

def applyANMS(corners, N_best):
    if len(corners) <= N_best:  # If there are N_best or fewer corners, just return them all
        return corners
    
	# Calculate the robustness of each corner
    robustness = np.zeros(len(corners))
    for i, corner_i in enumerate(corners):
        min_distance = np.inf
        for j, corner_j in enumerate(corners):
            if i != j:
                euclidean_distance = np.linalg.norm(np.array(corner_i) - np.array(corner_j))
                if euclidean_distance < min_distance:
                    min_distance = euclidean_distance
        robustness[i] = min_distance
    
	# Select the N_best corners with the highest robustness
    best_indices = np.argsort(-robustness)[:N_best]  # Sort robustness in descending order and take the first N_best
    anms_corners = corners[best_indices]
                
    return anms_corners


def extractFeatures(image, anms_corners):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    descriptors = []
    size = 41

    mean_img = np.mean(gray)
    std_img = np.std(gray)

    ymax, xmax = gray.shape

    # Define the size of the patch for each keypoint
    half_size = size // 2

    for point in anms_corners:
        x, y = point.ravel()
        # # Ensure the patch doesn't go outside the image boundaries
        # if x - half_size >= 0 and x + half_size < gray.shape[1] and y - half_size >= 0 and y + half_size < gray.shape[0]:
        xlower = max(0, x - half_size)
        xupper = min(xmax, x + half_size)
        ylower = max(0, y - half_size)
        yupper = min(ymax, y + half_size)
        
        # Extract the patch
        # patch = gray[y - half_size:y + half_size + 1, x - half_size:x + half_size + 1]
        patch = gray[ylower:yupper + 1, xlower:xupper + 1]
        patch = (patch - mean_img) / std_img

        # Apply Gaussian Blur
        patch_blurred = cv2.GaussianBlur(patch, (5, 5), 1)
        
        
        # Resize (subsample) the blurred patch
        patch_resized = cv2.resize(patch_blurred, (8,8))
        
        # Reshape to a 64x1 vector
        feature_vector = patch_resized.flatten()
        # print(feature_vector.shape)

        # Standardize the vector
        mean = np.mean(feature_vector)
        std = np.std(feature_vector)
        feature_vector = (feature_vector - mean) / (std + 1e-10)  # Adding a small constant to avoid division by zero
        
        descriptors.append(feature_vector)

    return np.array(descriptors)

def matchFeatures(descriptors1, descriptors2):
    ratio_threshold = 0.90
    matches = []
    # Iterate over all descriptors in the first image
    for idx1, descriptor1 in enumerate(descriptors1):
        distances = []
        # Compare with all descriptors in the second image
        for descriptor2 in descriptors2:
            ssd = np.sum((descriptor1 - descriptor2) ** 2)
            distances.append(ssd)

        # Sort matches based on SSD
        sorted_distances_idx = np.argsort(distances)
        # Take the ratio of the best match to the second best match
        if distances[sorted_distances_idx[0]] < ratio_threshold * distances[sorted_distances_idx[1]]:
            # If ratio test passes, add to good matches
            matches.append(cv2.DMatch(idx1, sorted_distances_idx[0], distances[sorted_distances_idx[0]]))

    return matches

def removeOutliers(keypoints1, keypoints2, matches):
    if len(matches) < 4:
        raise ValueError("Not enough points to compute homography.")
    
    # Convert keypoints to numpy arrays
    points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
    points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])

    threshold = 5
    confidence = 0.99
    max_iterations = 5000
    
    max_inliers = 0
    best_homography = None
    best_inliers = []
    
    for _ in range(max_iterations):
        # Randomly select 4 matches
        sample_indices = np.random.choice(len(matches), 4, replace=False)
        sample_points1 = points1[sample_indices]
        sample_points2 = points2[sample_indices]

        H, _ = cv2.findHomography(sample_points1, sample_points2, method=0)  # method=0 means using all points, no RANSAC yet

        # If homography could not be computed, skip this iteration
        if H is None:
            continue
        
        # Apply homography to all points in the first image
        transformed_points1 = cv2.perspectiveTransform(np.array([points1]), H)
        transformed_points1 = transformed_points1[0]

        # Compute SSD (Sum of Squared Differences) between transformed points and points in the second image
        ssd = np.sum((points2 - transformed_points1) ** 2, axis=1)

        # Determine inliers (where SSD is below the threshold)
        inlier_indices = np.where(ssd < threshold ** 2)[0]
        num_inliers = len(inlier_indices)

        # Update best homography if the current one has more inliers
        if num_inliers > max_inliers:
            max_inliers = num_inliers
            best_homography = H
            best_inliers = inlier_indices
        
        # Check if we have enough inliers or if we've reached the desired confidence level
        inlier_ratio = num_inliers / len(matches)
        if inlier_ratio > confidence:
            break

    # Re-compute least-squares homography estimate on all of the inliers
    if best_inliers.size > 0:
        inlier_points1 = points1[best_inliers]
        inlier_points2 = points2[best_inliers]
        best_homography, _ = cv2.findHomography(inlier_points1, inlier_points2, method=0)

    # Filter matches to only keep inliers
    inlier_matches = [matches[i] for i in best_inliers]

    # print(best_homography)
    return inlier_matches, best_homography

def warpAndBlend(image1, image2, homography):

    height1, width1 = image1.shape[:2]
    height2, width2 = image2.shape[:2]
    
    corners1 = np.float32([[0, 0], [0, height1], [width1, height1], [width1, 0]]).reshape(-1, 1, 2)
    # Apply homography to the corners of image1
    warped_corners1 = cv2.perspectiveTransform(corners1, homography)
    
    corners2 = np.float32([[0, 0], [0, height2], [width2, height2], [width2, 0]]).reshape(-1, 1, 2)

    # Combine the corners of image1 (warped) and image2
    all_corners = np.concatenate((warped_corners1, corners2), axis=0)

    [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

    # Translation matrix for adjusting the homography
    translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float32)

    # Warp image1 using the adjusted homography
    warped_image1 = cv2.warpPerspective(image1, translation.dot(homography), (x_max - x_min, y_max - y_min))




    # # Translate image2 to the new canvas
    # translated_image2 = cv2.warpPerspective(image2, translation, (x_max - x_min, y_max - y_min))

    # # Create a mask to identify where to merge image1 and image2
    # mask1 = np.zeros((y_max - y_min, x_max - x_min), dtype=np.uint8)
    # mask1[:height1, :width1] = 255

    # # Blend the two images
    # panorama = np.where(mask1[..., None], warped_image1, translated_image2)


    panorama = warped_image1.copy()
    panorama[-y_min:-y_min+height2, -x_min: -x_min+width2] = image2

    indices = np.where(image2 == [0,0,0])
    y = indices[0] + -y_min 
    x = indices[1] + -x_min 

    panorama[y,x] = warped_image1[y,x]

    return panorama

def stitchImages(images):
    # Assuming the first image is the base for the panorama
    panorama = images[0]
    
    # Iterate over all image pairs
    for i in range(1, len(images)):
        corners1 = detectCorners(panorama)
        corners1_vis = corners1.copy()
        corners1_vis = corners1.reshape(-1, 2)
        visualizeAndSaveCorners(panorama, corners1_vis, "Phase1/Code/Results/corners1.png")

        anms_corners1 = applyANMS(corners1, N_best = 1000)
        anms_corners1_vis = anms_corners1.copy()
        anms_corners1_vis = anms_corners1.reshape(-1, 2)
        visualizeAndSaveCorners(panorama, anms_corners1_vis, "Phase1/Code/Results/anms1.png")

        descriptors1 = extractFeatures(panorama, anms_corners1)
        # print(descriptors1)

        corners2 = detectCorners(images[i])
        corners2_vis = corners2.copy()
        corners2_vis = corners2.reshape(-1, 2)
        visualizeAndSaveCorners(images[i], corners2_vis, "Phase1/Code/Results/corners2.png")

        anms_corners2 = applyANMS(corners2, N_best = 1000)
        anms_corners2_vis = anms_corners2.copy()
        anms_corners2_vis = anms_corners2.reshape(-1, 2)
        visualizeAndSaveCorners(images[i], anms_corners2_vis, "Phase1/Code/Results/anms2.png")

        descriptors2 = extractFeatures(images[i], anms_corners2)

        # Convert corners to KeyPoint objects for visualization
        keypoints1 = [cv2.KeyPoint(x=float(pt[0][0]), y=float(pt[0][1]), size=1) for pt in anms_corners1]
        keypoints2 = [cv2.KeyPoint(x=float(pt[0][0]), y=float(pt[0][1]), size=1) for pt in anms_corners2]

        matches = matchFeatures(descriptors1, descriptors2)
        # Visualize and save the matches
        visualizeAndSaveMatches(panorama, keypoints1, images[i], keypoints2, matches, 'Phase1/Code/Results/matching.png')
        
        inliers, homography = removeOutliers(keypoints1, keypoints2, matches)

        visualizeAndSaveMatches(panorama, keypoints1, images[i], keypoints2, inliers, 'Phase1/Code/Results/matching_ransac.png')

        # Step 7: Warp and Blend
        panorama = warpAndBlend(panorama, images[i], homography)

    cv2.imshow("Panorama", panorama)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    cv2.imwrite('Phase1/Code/Results/mypano.png',panorama)
            
    return panorama

        


def main():
    # Add any Command Line arguments here
    # Parser = argparse.ArgumentParser()
    # Parser.add_argument('--NumFeatures', default=100, help='Number of best features to extract from each image, Default:100')

    # Args = Parser.parse_args()
    # NumFeatures = Args.NumFeatures
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--ImagesFolder', default='./Phase1/Data/Train/Set1/', help='Directory of images')
    args = parser.parse_args()
    images_folder = args.ImagesFolder

    images = readImages(images_folder)

    """
    Read a set of images for Panorama stitching
    """
    images = readImages(images_folder)
    # print(len(images))
    
    # for idx, image in enumerate(images):
    #     # Detect corners
    #     corners = detectCorners(image)
    #     # Apply ANMS
    #     best_corners = applyANMS(corners, N_best=500)
	# 	# Reshape corners for visualization
    #     corners = corners.reshape(-1, 2)
    #     best_corners = best_corners.reshape(-1, 2)
	# 	# Visualize and save original corners
    #     visualizeAndSaveCorners(image, corners, f'/Phase1/Code/Results/corners_{idx}.png')
	# 	# Visualize and save ANMS corners
    #     visualizeAndSaveCorners(image, best_corners, f'/Phase1/Code/Results/anms_{idx}.png')

    panorama = stitchImages(images)

    """
	Corner Detection
	Save Corner detection output as corners.png
	"""

    """
	Perform ANMS: Adaptive Non-Maximal Suppression
	Save ANMS output as anms.png
	"""

    """
	Feature Descriptors
	Save Feature Descriptor output as FD.png
	"""

    """
	Feature Matching
	Save Feature Matching output as matching.png
	"""

    """
	Refine: RANSAC, Estimate Homography
	"""

    """
	Image Warping + Blending
	Save Panorama output as mypano.png
	"""


if __name__ == "__main__":
    main()
