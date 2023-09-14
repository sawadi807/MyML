import os
import numpy as np
import tensorflow
from PIL import Image
from keras.applications import InceptionResNetV2
from keras.models import Sequential, load_model
import splitter
from socket_client_SC import *

"""
head 모델의 추론 결과인 intermediate data와 추론의 정답인 label_list를 socket_client_SC코드를 이용해 server로 전송하는 코드
"""
input_list = []
label_list = []

# 학습된 가중치를 위해 trained_model load
trained_model = load_model("SC_best_model_InceptionResNetV2.h5")
trained_weight = trained_model.get_weights()
# print(len(trained_weight))

# 모델 불러오기 및 분할
conv_base = InceptionResNetV2(
    weights="imagenet", include_top=False, input_shape=(150, 150, 3)
)
head_model, tail_model = splitter.split_model(conv_base, "block8_1")

head = Sequential()
head.add(head_model)
head.set_weights(trained_weight[0:730])
# print(len(head.get_weights()))

# 폴더 경로 지정
test_folder_path = r"practice_val_test"

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
intermediate_data = head.predict(input_list)

socket_client(intermediate_data)
socket_client(label_list)
