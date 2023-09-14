import os
import numpy as np
import tensorflow
from PIL import Image

# from tensorlfow.keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
from evaluation_util import evaluation_for_SC

input_list = []
label_list = []

# 모델 불러오기
model = load_model("img_best_model_InceptionResNetV2val.h5")

# 폴더 경로 지정
test_folder_path = r"C:\project\ISIC_new\test"

# 폴더 내의 파일들을 set화
label_num = -1
for label in os.listdir(test_folder_path):
    label_num += 1
    for filename in os.listdir(os.path.join(test_folder_path, label)):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            # 이미지 불러오기 및 전처리
            image_path = os.path.join(test_folder_path, label, filename)
            image = Image.open(image_path)
            image = image.resize((150, 150))
            image = np.array(image)
            image = image.astype("float32") / 255.0
            input_list.append(image)
            label_list.append(label_num)
input_list = np.array(input_list)
label_list = np.array(label_list)

# 예측
inference_result = model.predict(input_list)

benign = 0
malignant = 0
for i in inference_result:
    if np.argmax(i) == 0:
        benign += 1
    else:
        malignant += 1
print(benign, malignant)

# 정확도 출력
print("Accuracy = ", evaluation_for_SC(inference_result, label_list))

# predicted_class = np.argmax(prediction)

# # 결과 출력
# if predicted_class == 0:
#     print("The image is classified as benign.")
#     benign += 1
# elif predicted_class == 1:
#     print("The image is classified as malignant.")
#     malignant += 1
# else:
#     print("Invalid prediction.")
