# Text-to-Image

Naver Cloud, 글로벌공학교육센터, 공학교육혁신센터에서 후원하는 'OUTTA AI 부트캠프'를 통해 진행한 프로젝트이다.

해당 모델은 텍스트를 입력 받으면 해당 텍스트에 부합하는 이미지를 만들어주는 기능을 한다.
본 모델은 google의 colab에서 진행되었다.


1) 모델 구조

해당 모델의 구조는 generator와 discriminator로 구분되어져 있다. 입력 데이터로는 이미지와 해당 이미지에 대한 텍스트 정보를 clip 알고리즘을 통해 추출한 특징 정보이다. 
Generator는 입력 받은 데이터를 통해 (64by64), (128by128), (256by256)의 해상도를 가진 이미지를 순차적으로 출력하며 각각의 이미지는 Discriminator의 입력으로 들어갈 수 있게 한다. 이때 Generator는 텍스트에 해당하는 이미지를 학습해 나가며 점점 더 실제와 같은 이미지를 학습할 수 있게 된다.
Discriminator는 입력으로 텍스트와 해당 텍스트로 Generator가 생성한 Fake image 그리고 Real image를 받게 된다. 그리고 조건의 여부에 따라 real image와 fake image의 유사정도를 판별하게 된다. 이를 통해 Discriminator는 점점 실제와 같아지는 Fake image로 인해 loss는 커진다는 특징이 있다. 해당 코드에서는 조건이 없을 때를 학습했다. 

![image](https://github.com/sawadi807/MyML/assets/139100722/bb39ebba-98b7-4f0f-a01d-9a9049974072)


2) 코드 설명

코드 상에 프로젝트로 진행하며 직접 구현한 파일은 network.py와 train.py로 이 둘의 설명을 적어놓았다.

Ⅰ. network.py


Create C_hat & image

1. `ConditioningAugmention`
이 클래스는 텍스트 임베딩을 조건화하는 과정을 담당한다. 특히, CLIP 텍스트 임베딩(c_txt)을 입력으로 받아 증강된 텍스트 임베딩을 생성한다. 
	초기화 함수(__init__): 입력 차원(input_dim), 임베딩 차원(emb_dim), 및 장치(device)를 입력으로 받아, 전체 레이어를 정의한다. 이 때, 선형 레이어와 ReLU 활성화 함수를 포함하는 nn.Sequential을 사용한다.
	순전파 함수(forward): 입력 x를 받아 증강된 텍스트 임베딩, 평균(mu), 및 로그 시그마(log_sigma)를 반환한다. mu와 log_sigma는 각각 self.layer 출력의 절반 길이를 가진다. 또한, 표준 정규 분포에서 추출된 z와 함께 mu와 log_sigma를 사용하여 condition을 계산한다. 이 과정은 특정 조건을 반영한 임베딩을 생성하는 데 사용된다.
2. `ImageExtractor`
이 클래스는 입력 텐서에서 이미지를 추출하는 역할을 한다.
	초기화 함수(__init__): 입력 채널 수(in_chans)를 받아 출력 채널 수(out_chans)를 3으로 설정한다. 이후, 전치 합성곱 레이어와 하이퍼볼릭 탄젠트 활성화 함수를 포함하는 nn.Sequential을 정의한다. 하이퍼볼릭 탄젠트는 이미지의 픽셀 값을 [-1, 1] 범위로 정규화하는 데 사용된다.
	순전파 함수(forward): 입력 텐서 x를 받아 self.image_net을 통과시켜 이미지를 추출하고 반환한다. 출력 이미지는 [3, H, W] 형태를 가진다.


Generator

Generator는 text embedding과 noise를 입력으로 받아 이미지를 생성하는 역할을 한다. 여러 개의 generator 클래스와 재귀적인 구조를 사용하여 복잡한 이미지 생성 과정을 단계적으로 수행하며 구조는 크게 한 개의 Generator_type1과 두 개의 Generator_type2로 이루어져 있다.

1. Generator_type1
Generator_type1는 condition과 noise를 입력받아 이미지를 생성하는 역할을 하며 mapping network, upsample layer, image extractor로 이루어져 있다. 
	우선 noise와 CANet으로부터 추출된 \hat{c}_txt인 condition을 concat한 후 이를 mapping_network에 통과시켜 (𝑁𝑔 * 4 * 4) 차원을 만들어 준다. 이때 mapping_network를 거치게 되면 [batch_size, 𝑁𝑔 * 4 * 4] shape을 갖는데 upsample layer에 input 될 수 있도록 [batch_size, 𝑁𝑔, 4, 4] 형태로 만들어 준다. 그 후 각 block을 통과할 때마다 height/width 를 각각 2배로, channel 수를 0.5배로 만드는 upsample layer를 통과하여 최종적으로 [Ng/16, 64, 64] shape을 가진 텐서인 out을 출력하게 된다. 또한 이를 앞서 구현한 Image Extractor에 input으로 넣어 [3,64,64] shape의 이미지 image_out을 출력하게 된다. 최종적으로 Generator_type1에서는 out과 image_out을 출력하게 된다.
2. Generator_type2
Generator_type2는 condition과 이전 단계의 출력을 받아서 이미지를 생성하는 역할을 하며 joining layer, res layer, upsample layer, image extractor로 이루어져 있다. 
	Condition과 이전 layer의 output인 prev_output이 concat 된 것이 generator_type2의 input으로 들어가게 된다. 이때 condition과 prev_out을 concat하기 위해서는 서로 shape이 같아야 하는데 condition의 shape은 [batch_size, 128]이고 prev_output의 shape은 [batch_size, C/2, 2H, 2W]이므로 reshape와 repeat을 통해 두 shape을 일치시킨 후 concat 해 generator_type2로 들어가는 combined input을 만들어준다. Combined input은 가장 먼저 joining layer에 들어가 입력 채널 수인 condition_dim이 in_chans로변경되게 된다. 그 후 gradient 소실 문제를 해결하기 위해 res layer를 거치고 이 값이 upsample layer의 input으로 들어가 [batch_size, C/2, 2H, 2W] shape를 가진 out이라는 tensor를 반환하게 된다. 또한 이를 앞서 구현한 Image Extractor에 input으로 넣어 [3, 2H, 2W] shape의 이미지 image_out을 출력하게 된다. 최종적으로 Generator_type2에서도 Generator_type1과 마찬가지로 out과 image_out 이 두가지를 출력하게 된다. 
3. Generator class
Generator class는 전체 generator network를 구성하며, text_embedding과 noise를 결합하여 앞서 구현한 generator_type1, generator_type2를 이용해 다양한 단계에서 이미지를 생성하는 복잡한 작업을 수행한다.
	우선 text embedding을 conditioning augmentation에 input으로 넣어 \hat{c}_txt, mu, log_sigma를 얻는다. 그 후 _stage_generator를 이용하여 첫 stage에서는 generator1을, 나머지 stage에서는 generator2를 반환하게 한다. 이후 generator stage를 차례대로 거치며 이미지 fake_image를 생성하고 이를 리스트 fake_images에 저장한다. 최종적으로 fake_images와 mu, log_sigma을 반환하게 된다.

4. `ResModule`
`ResModule`은 신경망 내에서 잔차 연결(residual connection)을 수행하는 모듈이다. 이는 딥 러닝 모델, 특히 컨볼루션 신경망(Convolutional Neural Network, CNN)에서 성능을 향상시키는 일반적인 기법 중 하나이다. 

	초기화 (__init__ 메서드) : ResModule은 입력 채널 크기 in_chans를 인자로 받아 클래스를 초기화한다. 이 값을 기반으로 모듈 내부의 컨볼루션과 배치 정규화 계층을 구성한다.
-	`nn.Conv2d`: 입력 채널 크기와 동일한 출력 채널 크기로 3x3 컨볼루션 필터를 적용한다. 패딩은 1로 설정하여 입력과 출력의 크기를 유지한다.
-	`nn.BatchNorm2d`: 배치 정규화를 적용하여 학습을 안정화하고 빠르게 수렴하도록 돕는다.
-	`nn.ReLU`: 활성화 함수로 ReLU를 사용하여 비선형성을 추가한다.

	순전파 (`forward` 메서드) : 순전파 단계에서는 입력 텐서 x를 받아 컨볼루션, 배치 정규화, ReLU 활성화 함수를 거친 후 원래 입력 텐서 x를 더한다. 이 구조를 수식으로 표현하면 res_out = x + processed_x와 같다. 이러한 잔차 연결은 신경망이 더 깊게 학습될 수 있게 돕는다. 결과적으로, ResModule은 입력과 동일한 형태의 출력 텐서 res_out을 반환한다. 
Generator_type2 내에서 ResModule은 gradient 소실 문제를 해결하기 위해 사용된다. 깊은 신경망에서는 역전파 과정 중에 그래디언트가 점차 작아지는 현상이 발생할 수 있는데, 이러한 문제를 해결하기 위해 ResModule을 도입한다.


Discriminator

Generator를 통해 최종적으로 완성된 fake image는 Dicriminator의 input으로 들어가게 된다. Discrminator는 받은 fake image와 real image 그리고 text를 통해 최종 결과를 도출해 낸다. 이때, fake image에는 조건을 추가해(CondDiscriminator) 정확도를 높이거나, fake image와 real image의 유사성을 판별한 후 이를 결과 도출에 이용해(AlignCondDiscriminator) 정확도를 높일 수도 있다. 
1.	UncondDiscriminator에서는 추가될 조건이 없기 때문에, 입력으로 [10, 8Nd, 4, 4]를 받으면 [10, 1]로 변환하고 있다. 이를 위해 한 개의 Conv2d(8 * in_chans, 1, kenel_size=4, stride=4, padding=0) + BN + LeakyReLU을 사용해 [10, 1, 1, 1]로 변환 후 Flatten을 통해 [10, 1]로 만든다.
2.	CondDiscriminator에서는 fake image에 조건이 추가된다. 추가되는 조건을 입력으로 받으면 forward 함수에서 view를 통해 [batch_size, condition, 1, 1]로 변환한 후, repeat을 통해 H와 W는 x와 일치시킨다. 이후, concat을 진행해 cond_layer의 입력으로 넣는다. 이때, 반환은 [10, 1]이 되어야 하므로, Conv2d(in_chas+condition_dim, 8*in_chans, kenel_size=4, stride=4, padding=0) + BN + LeakyReLU 를 사용해 채널 길이만 8Nd로 변환한 후에, 해당 값을 UncondDiscriminator와 동일한 방법으로 [10,1]을 반환한다.
3.	AlignCondDiscriminator에서는 aligning을 한 조건이 추가된다. CondDiscriminator과 과정은 동일하나 align_layer의 최종 반환의 채널은 clip_embedding_dim이 되게 한다.
마지막 Discriminator에서는 forward를 통해 이미지와 조건 여부를 입력으로 받은 후, Sigmoid를 통해 예측 확률을 도출해낸다. 받은 이미지는 [3, H, W]이므로 위에서 만든 3개의 class의 입력으로 사용하기 위해 [8Nd, H/16, H/16]으로 변환 후, [8Nd, H, H]으로 주어야 한다. 이는 차례대로 아래의 구조를 가진 _global_discriminator, _prior_layer을 통해 만들 수 있다. _prior_layer가 받는 입력은 만들어진 이미지 크기가 128인지 256인지에 따라 달라지므로 _prior_layer와 _prior_layer2로 구분하였다.
(Conv2d(img_chans, in_chans, kenel_size=4, stride=2, padding=1) + LeakyReLU) *  1  +
(Conv2d(이전_out_Dim, 2 * 이전_out_Dim , kenel_size=4, stride=2, padding=1) + BN + LeakyReLU) * 3
Prior_layer까지 거쳐 만들어진 output은 앞서 정의한 조건 여부에 따른 함수를 거치게 되고 최종 결과를 도출한다. 이때 결과의 out은 sigmoid 함수를 거쳐 0부터 1사이의 확률 값으로 나오게 된다.
마지막의 weight_init을 통해 사용한 layer를 초기화하고 있다.



Ⅱ. train.py

train.py코드는 GAN을 활용해서 이미지를 생성할 수 있는 모델을 학습할 수 있도록 구현한 코드이다.
●contrastive_loss_G에서는 생성된 가짜 이미지와 텍스트 임베딩 사이의 Contrastive Loss를 계산한다. 정규화된 형태의 이미지를 224로 바꾸고 원하는 형태로 재구성한 후에 clip모델에 입력할 수 있는 형태로 전처리한다. 그 후에 clip모델을 사용해서 이미지를 임베딩 벡터로 변환 및 정규화하고 두 벡터 사이의 코사인 유사성을 계산한 후에 Contrastive loss를 계산한다.
●contrastive_loss_D에서는 Discriminator가 생성된 이미지와 텍스트 임베딩 간의 Contrastive loss를 계산한다. 생성된 이미지에서 특징을 뽑아낸 후에 정규화하고 그것을 model_features에 저장한다. 그 후에 이미지의 특징과 텍스트 임베딩 사이에서의 코사인 유사성을 계산하고 contrastive loss를 계산한다.
●D_loss에서는 Discriminator의 Loss를 계산한다. 먼저 생성된 가짜 이미지와 실제 이미지의 Discriminator 출력을 계산한 후에 미리 정의되어 있는 레이블을 활용해서 가짜 이미지와 실제 이미지의 손실을 계산한다.
●G_loss에서는 Generator의 Loss를 계산한다. 먼저 생성된 가짜 이미지와 해당하는 Discriminator의 출력을 계산한 후에 미리 정의되어 있는 레이블을 사용해서 각각의 손실을 계산한다. 
●train_step함수는 학습 단계를 수행하는 함수이다. 먼저 데이터 로더에서 배치를 가져와서 실제 이미지들을 디바이스로 옮긴다. Discriminator와 Generator를 최적화하고 각각의 이미지와 레이블의 Discriminator와 Generator의 loss를 계산한다. Train_loader에서 배치 단위로 데이터를 불러오는데 실제 이미지와 이미지의 특징과 텍스트의 특징이 이 배치에 포함되어 있다. 현재의 스테이지의 Discriminator를 업데이트하고 실제 이미지와 가짜 이미지의 손실을 계산한 후에 손실로부터 Discriminator를 업데이트한다. 이런 방식을 통해서 단계별로 Discriminator와 Generator의 loss를 계산하고 epoch마다 loss와 이미지를 출력한다. 
●train은 실제로 학습을 진행하는 함수이다. train_loader에서 배치 단위로 데이터를 불러오는데 실제 이미지와 이미지의 특징, 텍스트의 특징이 이 배치에 포함되어 있다. 각 배치를 활용해서 Discriminator는 실제 이미지와 생성된 이미지를 구별하는 것을 학습하고 Generator는 생성된 이미지가 실제 이미지와 비슷하게 보일 수 있도록 한다. 학습된 Discriminator와 Generator의 출력을 바탕으로 각 epoch마다 loss를 출력하고 이미지를 저장한다. 학습이 끝난 후에는 Discriminator와 Generator의 loss값을 각각 출력한다. 


3) 결과

예시로 얻어진 결과는 아래와 같다. 시간에 제약이 있어 더 좋은 결과를 얻지 못한 점은 다소 아쉬웠다.

![잘나온이미지_전체](https://github.com/sawadi807/MyML/assets/139100722/915a71bf-b744-4281-8b71-1dc029616c36)




이들은 test를 위해 실제 텍스트를 입력했을 때 얻은 결과로 흐리지만 꽤 정확한 이미지가 나온 것을 볼 수 있다. 
"The woman is young and has black hair, and arched eyebrows." 에 대한 출력 결과이다.

![잘나온테스트1](https://github.com/sawadi807/MyML/assets/139100722/80ec9638-1beb-4990-9434-8ca4085019a2)![잘나온테스트2](https://github.com/sawadi807/MyML/assets/139100722/fdaf8048-735d-4c14-bbe2-d5410910bc21)







  
