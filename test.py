import os
import numpy as np
from PIL import Image
from keras.models import Sequential, load_model
from keras import layers, regularizers
from keras.applications import InceptionResNetV2
import splitter

# 모델 불러오기
trained_model = load_model("img_best_model_InceptionResNetV2.h5")
trained_weight = trained_model.get_weights()

# 모델 불러오기 및 분할
conv_base = InceptionResNetV2(
    weights="imagenet", include_top=False, input_shape=(150, 150, 3)
)
head_model, tail_model = splitter.split_model(conv_base, "block8_1")

head = Sequential()
head.add(head_model)
head.set_weights(trained_weight[0:730])

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

# 파일 경로 지정
test_file_path = r"practice_train\malignant\ISIC_0149568_0_1451.jpg"

# 이미지 불러와 예측
if test_file_path.lower().endswith((".png", ".jpg", ".jpeg")):
    image = Image.open(test_file_path)
    image = image.resize((150, 150))
    image = np.array(image)
    image = image.astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)

    # 예측
    intermediate_data = head.predict(image)
    inference_result = tail.predict(intermediate_data)
    predicted_class = np.argmax(inference_result)

    # 결과 출력
    print(inference_result)
    print(f"Image: {test_file_path}")
    if inference_result[0][1] > 0.9:
        print("해당 이미지는 악성 종양일 가능성이 있습니다.")
    else:
        print("해당 이미지는 정상입니다.")

else:
    print("Wrong Directory!")
    exit(0)
