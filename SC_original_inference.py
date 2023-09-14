import os
import numpy as np
import tensorflow
from PIL import Image
from keras import models, layers, regularizers
from keras.utils import plot_model
from keras.applications import InceptionResNetV2
from keras.models import Sequential, load_model
from evaluation_util import evaluation_for_SC
import splitter


#### split 하지 않은 convetional inference code

input_list = []
label_list = []

# 학습된 가중치를 위해 trained_model load
trained_model = load_model("SC_best_model_InceptionResNetV2.h5")
trained_weight = trained_model.get_weights()
# print(len(trained_weight))

# 모델 불러오기 및 분할
conv_base = InceptionResNetV2(
    weights="imagenet", include_top=False, input_shape=(224, 224, 3)
)
head_model, tail_model = splitter.split_model(conv_base, "block8_1")

head = Sequential()
head.add(head_model)
head.set_weights(trained_weight[0:730])
# print(len(head.get_weights()))

tail = Sequential()
tail.add(tail_model)
tail.add(layers.GlobalAveragePooling2D())
tail.add(layers.Dropout(0.3))
tail.add(
    layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(0.005))
)
tail.add(layers.Dropout(0.3))
tail.add(layers.Dense(2, activation="sigmoid"))
tail.set_weights(trained_weight[730:])
# print(len(tail.get_weights()))

# 폴더 경로 지정
test_folder_path = r"내 드라이브"

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
inference_result = tail.predict(intermediate_data)

print(type(intermediate_data))

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
