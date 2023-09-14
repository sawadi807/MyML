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
from socket_server_SC import *

"""
socket을 통해 client로부터 intermediate data와 label_list를 전송받는다.
intermediate data를 input으로 하여 tail 부분의 추론을 마친뒤 그 결과인 inference_result와
label_list를 비교하여 정확도를 나타낸다.
"""
# 학습된 가중치를 위해 trained_model load
trained_model = load_model("SC_best_model_InceptionResNetV2.h5")
trained_weight = trained_model.get_weights()
# print(len(trained_weight))

# 모델 불러오기 및 분할
conv_base = InceptionResNetV2(
    weights="imagenet", include_top=False, input_shape=(150, 150, 3)
)
head_model, tail_model = splitter.split_model(conv_base, "block8_1")

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


# 예측
intermediate_result = socket_server()
label_list = socket_server()

inference_result = tail.predict(intermediate_result)

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
