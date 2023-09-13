from keras import layers
from keras import Input
from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import BatchNormalization, Conv2D, Activation, Dense, GlobalAveragePooling2D, MaxPooling2D, ZeroPadding2D, Add
from keras.optimizers import Adam
 

# 데이터 경로
train_dir = r'E:\대학_(2023)\졸논\최종데이터셋\학습'
val_dir = r'E:\대학_(2023)\졸논\최종데이터셋\검증'

# 이미지 크기와 배치 크기 설정
img_width, img_height = 224, 224
batch_size = 16

# 데이터 증강(Augmentation) 설정
train_datagen = ImageDataGenerator(rescale=1./255,
                                    rotation_range=20,        # 1번: 20도까지 임의 회전
                                    # width_shift_range=0.2,    # 2번: 너비 방향으로 최대 20%만큼 이동
                                    # height_shift_range=0.2,   # 2번: 높이 방향으로 최대 20%만큼 이동
                                    zoom_range=0.2,           # 4번: 최대 20%만큼 확대/축소
                                    horizontal_flip=True,     # 5번: 수평으로 뒤집기
                                    vertical_flip=True,       # 6번: 수직으로 뒤집기
                                    )

val_datagen = ImageDataGenerator(rescale=1./255,
                                rotation_range=20,        # 1번: 20도까지 임의 회전
                                # width_shift_range=0.2,    # 2번: 너비 방향으로 최대 20%만큼 이동
                                # height_shift_range=0.2,   # 2번: 높이 방향으로 최대 20%만큼 이동
                                zoom_range=0.2,           # 4번: 최대 20%만큼 확대/축소
                                horizontal_flip=True,     # 5번: 수평으로 뒤집기
                                vertical_flip=True,       # 6번: 수직으로 뒤집기
                                )


# 데이터 로딩 및 증강(Augmentation)
train_ds = train_datagen.flow_from_directory(train_dir,
                                             target_size=(img_width, img_height),
                                             batch_size=batch_size,
                                             class_mode='categorical')

val_ds = val_datagen.flow_from_directory(val_dir,
                                         target_size=(img_width, img_height),
                                         batch_size=batch_size,
                                         class_mode='categorical')


K = 2
 
 
input_tensor = Input(shape=(224, 224, 3), dtype='float32', name='input')
 
 
def conv1_layer(x):    
    x = ZeroPadding2D(padding=(3, 3))(x)
    x = Conv2D(32, (7, 7), strides=(2, 2))(x)
        
    x = Activation('relu')(x)
    x = ZeroPadding2D(padding=(1,1))(x)
 
    return x   
 
    
 
def conv2_layer(x):         
    x = MaxPooling2D((3, 3), 2)(x)     
 
    shortcut = x
 
    for i in range(3):
        if (i == 0):
            x = Conv2D(32, (1, 1), strides=(1, 1), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            
            x = Conv2D(32, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
 
            x = Conv2D(64, (1, 1), strides=(1, 1), padding='valid')(x)
            shortcut = Conv2D(64, (1, 1), strides=(1, 1), padding='valid')(shortcut)            
            x = BatchNormalization()(x)
            shortcut = BatchNormalization()(shortcut)
 
            x = Add()([x, shortcut])
            x = Activation('relu')(x)
            
            shortcut = x
 
        else:
            x = Conv2D(32, (1, 1), strides=(1, 1), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            
            x = Conv2D(32, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
 
            x = Conv2D(64, (1, 1), strides=(1, 1), padding='valid')(x)
            x = BatchNormalization()(x)            
 
            x = Add()([x, shortcut])   
            x = Activation('relu')(x)  
 
            shortcut = x        
    
    return x
 
 
 
def conv3_layer(x):        
    shortcut = x    
    
    for i in range(4):     
        if(i == 0):            
            x = Conv2D(64, (1, 1), strides=(2, 2), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)        
            
            x = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)  
 
            x = Conv2D(256, (1, 1), strides=(1, 1), padding='valid')(x)
            shortcut = Conv2D(256, (1, 1), strides=(2, 2), padding='valid')(shortcut)
            x = BatchNormalization()(x)
            shortcut = BatchNormalization()(shortcut)            
 
            x = Add()([x, shortcut])    
            x = Activation('relu')(x)    
 
            shortcut = x              
        
        else:
            x = Conv2D(64, (1, 1), strides=(1, 1), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            
            x = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
 
            x = Conv2D(256, (1, 1), strides=(1, 1), padding='valid')(x)
            x = BatchNormalization()(x)            
 
            x = Add()([x, shortcut])     
            x = Activation('relu')(x)
 
            shortcut = x      
            
    return x
 
 
 
def conv4_layer(x):
    shortcut = x        
  
    for i in range(6):     
        if(i == 0):            
            x = Conv2D(256, (1, 1), strides=(2, 2), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)        
            
            x = Conv2D(256, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)  
 
            x = Conv2D(512, (1, 1), strides=(1, 1), padding='valid')(x)
            shortcut = Conv2D(512, (1, 1), strides=(2, 2), padding='valid')(shortcut)
            x = BatchNormalization()(x)
            shortcut = BatchNormalization()(shortcut)
 
            x = Add()([x, shortcut]) 
            x = Activation('relu')(x)
 
            shortcut = x               
        
        else:
            x = Conv2D(256, (1, 1), strides=(1, 1), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            
            x = Conv2D(256, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
 
            x = Conv2D(512, (1, 1), strides=(1, 1), padding='valid')(x)
            x = BatchNormalization()(x)            
 
            x = Add()([x, shortcut])    
            x = Activation('relu')(x)
 
            shortcut = x      
 
    return x
 
 
 
def conv5_layer(x):
    shortcut = x    
  
    for i in range(3):     
        if(i == 0):            
            x = Conv2D(512, (1, 1), strides=(2, 2), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)        
            
            x = Conv2D(512, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)  
 
            x = Conv2D(1024, (1, 1), strides=(1, 1), padding='valid')(x)
            shortcut = Conv2D(1024, (1, 1), strides=(2, 2), padding='valid')(shortcut)
            x = BatchNormalization()(x)
            shortcut = BatchNormalization()(shortcut)            
 
            x = Add()([x, shortcut])  
            x = Activation('relu')(x)      
 
            shortcut = x               
        
        else:
            x = Conv2D(512, (1, 1), strides=(1, 1), padding='valid')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            
            x = Conv2D(512, (3, 3), strides=(1, 1), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
 
            x = Conv2D(1024, (1, 1), strides=(1, 1), padding='valid')(x)
            x = BatchNormalization()(x)           
            
            x = Add()([x, shortcut]) 
            x = Activation('relu')(x)       
 
            shortcut = x                  
 
    return x


 
x = conv1_layer(input_tensor)
x = conv2_layer(x)
x = layers.Dropout(0.5)(x)
x = conv3_layer(x)
x = layers.Dropout(0.5)(x)
x = conv4_layer(x)
x = layers.Dropout(0.5)(x)
x = conv5_layer(x)
 
x = GlobalAveragePooling2D()(x)
output_tensor = Dense(K, activation='softmax', kernel_regularizer=regularizers.l2(0.004))(x)

 
model = Model(input_tensor, output_tensor)

model.compile(optimizer=Adam(learning_rate=0.00655),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# history = model.fit(train_ds,
#                     steps_per_epoch=train_ds.samples // train_ds.batch_size,
#                     epochs=3,
#                     validation_data=val_ds,
#                     validation_steps=val_ds.samples // val_ds.batch_size,
#                     verbose=1)

# model.save("만든_model")

# ModelCheckpoint 콜백 생성
checkpoint = ModelCheckpoint('best_model.h5',  # 저장될 모델 파일의 이름
                             monitor='val_accuracy',  # 모니터링할 지표 (여기서는 검증 정확도)
                             save_best_only=True,  # 가장 좋은 성능의 모델만 저장
                             mode='max',  # 지표를 최대화하는 방향으로 저장
                             verbose=1)  # 저장 시 메시지 출력

# 모델 훈련 시에 ModelCheckpoint 콜백 추가
model.fit(train_ds,
          steps_per_epoch=train_ds.samples // train_ds.batch_size,
          epochs=5,
          validation_data=val_ds,
          validation_steps=val_ds.samples // val_ds.batch_size,
          callbacks=[checkpoint])