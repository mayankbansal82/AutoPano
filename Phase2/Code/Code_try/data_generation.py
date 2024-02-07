import cv2
import numpy as np

def generate_data(image_index, patch_index, img, homography_list, corner_pointA):
    h, w, _ = img.shape
    patch_size = 128
    max_perturbation = 32

    # Choose random top-left corner of the patch within the image
    x, y = np.random.randint(0, w - patch_size), np.random.randint(0, h - patch_size)
    x1, y1 = x + patch_size, y + patch_size

    # Define the corner points of the patch
    corner_points = np.array([[x, y], [x, y1], [x1, y], [x1, y1]])
    corner_pointA.append(corner_points)

    # Apply random perturbation to the corner points
    perturbation = np.random.randint(-max_perturbation, max_perturbation + 1, (4, 2))
    perturbed_corners = corner_points + perturbation

    # Calculate homography and warp the image
    H = np.linalg.inv(cv2.getPerspectiveTransform(np.float32(corner_points), np.float32(perturbed_corners)))
    warped_img = cv2.warpPerspective(img, H, (w, h))

    # Extract patches from the original and warped images
    original_patch = img[y:y1, x:x1]
    warped_patch = warped_img[y:y1, x:x1]

    # Calculate the homography difference for the patches
    H4pt = (perturbed_corners - corner_points).astype(np.float32)
    homography_list.append(H4pt)

    # Save the patches as images
    cv2.imwrite(f'../Data/modified_val/patchA/{image_index}_{patch_index+1}.jpg', original_patch)
    cv2.imwrite(f'../Data/modified_val/patchB/{image_index}_{patch_index+1}.jpg', warped_patch)

    return homography_list, corner_pointA

def main():
    homography_list, corner_pointA = [], []

    for i in range(1, 1001):
        img = cv2.imread(f"../Data/Val/{i}.jpg")
        for j in range(3):
            homography_list, corner_pointA = generate_data(i, j, img, homography_list, corner_pointA)

    # Save homography and corner points data
    np.save('./TxtFiles/modified_val_labels.npy', np.array(homography_list))
    np.save('./TxtFiles/val_cornerpoints.npy', np.array(corner_pointA))

main()
