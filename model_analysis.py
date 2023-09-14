from tensorflow import keras
from keras import layers, regularizers
from keras.models import load_model, Sequential
from keras.utils import plot_model
from keras.applications import InceptionResNetV2
import splitter

# 모델 불러오기 및 분할
conv_base = InceptionResNetV2(
    weights="imagenet", include_top=False, input_shape=(150, 150, 3)
)
head_model, tail_model = splitter.split_model(conv_base, "block8_1")

head = Sequential()
head.add(head_model)
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
# print(len(tail.get_weights()))


plot_model(head, "head_structure.png", show_shapes=True, show_layer_names=True)
plot_model(tail, "tail_structure.png", show_shapes=True, show_layer_names=True)

original = Sequential()
original.add(conv_base)
original.add(layers.GlobalAveragePooling2D())
original.add(layers.Dropout(0.3))
original.add(
    layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(0.005))
)
original.add(layers.Dropout(0.3))
original.add(layers.Dense(2, activation="sigmoid"))
plot_model(original, "original_structure.png", show_shapes=True, show_layer_names=True)
