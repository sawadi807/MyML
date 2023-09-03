# from PIL import Image
# import os
# import numpy as np
# from keras.preprocessing.image import ImageDataGenerator

# # 이미지 파일 경로
# image_path = 'AUG_0_0.jpeg'
# image_path2 = r'E:\증강'

# # 데이터 증강(Augmentation) 설정
# train_datagen = ImageDataGenerator(rescale=1./255,
#                                     rotation_range=20,        # 1번: 20도까지 임의 회전
#                                     # width_shift_range=0.2,    # 2번: 너비 방향으로 최대 20%만큼 이동
#                                     # height_shift_range=0.2,   # 2번: 높이 방향으로 최대 20%만큼 이동
#                                     zoom_range=0.2,           # 4번: 최대 20%만큼 확대/축소
#                                     horizontal_flip=True,     # 5번: 수평으로 뒤집기
#                                     vertical_flip=True,       # 6번: 수직으로 뒤집기
#                                     )

# # 이미지 로드
# image = Image.open(image_path)

# # 이미지를 배열로 변환
# image_array = np.array(image)

# # 배열 형태를 변경하여 배치 차원을 추가
# image_array = np.expand_dims(image_array, axis=0)

# # 변형된 이미지를 생성하여 저장
# augmented_images = train_datagen.flow(image_array, batch_size=1, save_to_dir=image_path2, save_prefix="augmented", save_format="jpeg")

# # 10장의 이미지를 생성하고 저장
# for i in range(10):
#     augmented_image = next(augmented_images)


import os
import cv2

# Set the directories for the input and output images
src_dir = r'E:\gray'
dst_dir2 = r"E:\gene"

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