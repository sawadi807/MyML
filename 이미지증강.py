# import os
# import cv2
# import numpy as np

# # # Define the augmentation functions
# # def contrast_stretching(img):
# #     img_rescale = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
# #     return img_rescale

# def adaptive_histogram_equalization(img):
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#     img_clahe = clahe.apply(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
#     return img_clahe

# # # CLAHE (Contrast Limited Adaptive Histogram Equalization)
# # def clahe(img):
# #     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# #     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
# #     img_clahe = clahe.apply(gray)
# #     return img_clahe


# # Set the directories for the input and output images
# src_dir = r"E:\train"
# dst_dir = r"E:\train_aug"

# # Get the list of image files in the source directory
# img_files = os.listdir(src_dir)

# # Process each image in the source directory
# for file_name in img_files:
#     # Read the image file
#     img = cv2.imread(os.path.join(src_dir, file_name), cv2.IMREAD_COLOR)

#     # Apply the augmentation functions
#     # img_contrast = contrast_stretching(img)
#     img_adapthisteq = adaptive_histogram_equalization(img)
#     # img_clahe = clahe(img)

#     # Save the augmented images
#     # cv2.imwrite(os.path.join(dst_dir, f"aug_contrast_{file_name}"), img_contrast)
#     cv2.imwrite(os.path.join(dst_dir, f"aug_adapthisteq_{file_name}"), img_adapthisteq)
#     # cv2.imwrite(os.path.join(dst_dir, f"aug_clahe_{file_name}"), img_clahe)





# 이미지 흑백 변환


import os
import cv2

# Set the directories for the input and output images
src_dir = r"E:\before"
dst_dir2 = r"E:\gray"

# Get the list of image files in the source directory
img_files = os.listdir(src_dir)

# Process each image in the source directory
for file_name in img_files:
    # Read the image file
    img = cv2.imread(os.path.join(src_dir, file_name))

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Save the grayscale image
    cv2.imwrite(os.path.join(dst_dir2, file_name), gray)
