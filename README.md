# Find-Cancer

경희대학교에서 4-1학기에 졸업 프로젝트로 진행한 작품이다.

해당 모델은 피부암을 진단하기 위해 아이폰의 접사기능을 이용해 사람의 점을 근접 촬영한 후 모델의 입력으로 넣는다. 나오는 결과는 해당 이미지가 점인지 피부암인지를 판별해준다. 

프로젝트의 모델 구현은 나, 소켓 프로그래밍은 팀원이 맡게 되었다.




1) 프로젝트 설계 배경 및 내용

피부암은 일반인이 보기에 정상적인 점과 구분이 쉽지 않다. 그래서 피부암은 방치되어 조기 진단을 놓치는 경우가 많다. 그렇다고 피부암이 의심될 때마다 병원에 가는 것은 환자의 입장에서는 소모적인 일이다. 따라서 mobile device로 간편하게 피부암을 진단할 수 있는 피부암 진단 인공지능이 필요하다.
한편, 복잡한 DNN은 스마트폰과 같은 mobile device에서 수행하기에 computing power가 부족할 수 있다. 그래서 일반적으로 mobile devcie에서 input data를 server로 전송하여 server에서 DNN 추론을 수행하는 클라우드 컴퓨팅 방법이 사용된다. 하지만 피부암 진단 DNN과 같은 의료 분야의 인공지능은 input data로 환자의 개인정보나 환부의 사진이 요구되기 때문에 전통적인 클라우드 컴퓨팅 방법은 data 유출 문제에 취약할 수 있다. 따라서 전통적인 cloud의 보안적 한계를 극복하는 피부암 진단 DNN 모델의 구현이 필요하다.





2) 프로젝트 진행 내용

① 데이터 수집 및 전처리
  - ISIC와 Kaggle을 통해 악성종양과 양성종양인 이미지를 각각 16023장과 2521장을 수집함.
  - 데이터 불균형 처리를 위해 증강을 통해 균형을 맞춤. 학습과 검증을 위한 이미지는 image generator를 사용해 random 이미지를 생성함. 
  - 테스트를 위한 이미지는 원본의 손상이 발생하면 안되므로 증강없이 원본 이미지의 일부를 준비함.
- 각각의 데이터는 악성종양과 양성종양인 이미지를 각각 학습에 12000장씩, 검증에 3000장씩, 테스트에 1023장과 341장을 사용함.

② 피부암 진단 DNN 구현
  - 데이터는 image generator을 통해 1/255 scale로 축소한 후, 최종 입력으로는 (150, 150)을 사용함. Batch size는 64로 설정했고 input shape는 (None, 150, 150, 3)이 됨.
  - 복잡도가 큰 모델이므로 속도를 높이기 위해 병렬 처리 기법을 사용함. 
  - 이미지 분류 모델인 ‘InceptionResNetV2’을 사용함. 불러온 모델의 최종 layer는 사용하지 않고 나온 결과를 GlobalAveragePooling2D를 통해 2차원으로 축소 시킨 후, dropout과 Dense layer를 거쳐 이진 분류를 위한 결과값을 출력함.
  - 만들어진 모델은 20번의 epoch를 시도함. 이때, 검증 정확도가 가장 높은 모델만 저장함. 
  - 학습 과정에서 learning rate, dropout rate, batch size 등과 같은 모델 parameter 값을 바꿔가며 최적의 값을 찾음.

③ split computing 구현
  - InceptionResNetV2는 여러 층의 복잡한 layer로 구성된 이미지 분류 모델이다. client와 server 각각에 모델을 할당하는 split computing을 구현하기 위해서 InceptionResNetV2 모델을 head와 tail 두 부분으로 나누어야 한다. 이를 위해 특정 layer를 기준으로 모델을 두 부분으로 나누어 반환하는 splitter.py 를 구현했다.
  - InceptionResNetV2를 head와 tail로 나눈다. 모델의 추론 결과는 이미지가 악성 종양인지 양성 종양인지 확인하는 것이기 때문에 tail 부분에는 이를 위한 layer를 추가한다.
  - client에 할당하는 head 모델은 이미지 파일을 input으로 하고 output으로 다차원 numpy array 타입의 intermediate data를 반환한다.
  - server에 할당하는 tail 모델은 head 모델의 output인 intermediate data를 input으로 하고 output으로 양성종양인 확률과 악성종양인 확률의 배열을 반환한다.

④ socket 통신 구현
  - client의 추론 결과인 intermediate data를 sever로 전송하기 위해서 TCP/IP 방식으로 통신하는 socket_client_SC.py와 socket_server_SC.py를 구현했다.
  - socket_client_SC.py 코드는 sever와 연결하여 serialize 된 다차원 numpy array를 전송한다.
  - socket_server_SC.py 코드는 client의 TCP 연결을 수용하여 serialized data를 받아 unserialize하여 본래의 numpy array로 복구한다.





3) 모델 구조

전체적인 모델 구조는 아래와 같다. 이 과정에서 InceptionResNetV2의 중간 부분을 분할하여 head와 tail로 분리해 split computing을 진행하였다.

![image](https://github.com/sawadi807/MyML/assets/139100722/fb1d4fae-cf81-4914-a6db-9142937de16e)



4) 결론


최종 포스터 : https://drive.google.com/file/d/1AduKRu3y3HG7opS96pq6CWhB_0STLH5R/view?usp=drive_link


 기존의 cloud computing 방식의 DNN 학습은 민감할 수 있는 환자의 개인정보를 외부로 유출해야한다는 단점이 있었다. 본 졸업 연구에서는 이러한 한계를 해결하고자 원본 데이터를 외부로 직접적으로 유출하지 않는 split computing 방식으로 피부암 진단 DNN을 구현하였다. 만약 intermediate data가 유출되더라도 식별이 어려운 다차원 배열 형식이기 때문에 보안적 측면에서 우수하다는 장점을 가진다. 이러한 점에서 split computing이 가지는 의의를 확인할 수 있었다.
 하지만 본 졸업 작품에서는 split computing 자체는 성공적으로 구현했지만 피부암 진단의 DNN의 정확도를 만족할 만큼 이끌어내지 못했다는 한계가 있다. 훈련과 검증 정확도는 높은 반면, 테스트에서는 이에 못미치는 정확도를 얻었다. 가장 큰 원인은 overfitting 문제를 완벽히 해결하지 못했기 때문이다. Overfitting의 근본적인 문제는 우선 데이터의 부족이다. 증강을 통해 부족한 데이터의 양을 늘려 문제를 해결하고자 했지만 어디까지나 인위적으로 증강한 이미지이기 때문에 DNN이 피부암의 특징을 제대로 학습하기에는 부족했다. 즉, 학습 데이터가 부족한 상황에서 이를 과도하게 학습하는 overfitting이 발생해 특징을 제대로 추출하지 못한 것이다. 향후 연구에서 이 문제를 해결하기 위해서는 overfitting이 일어나는 것을 방지할 수 있는 증강 기법이나 환자의 나이, 성별, 기타 질병 등을 메타 데이터로 추가하여 학습하는 방법을 제안할 수 있을 것이다.


모델 결과는 아래와 같다.

![image](https://github.com/sawadi807/MyML/assets/139100722/cae529ed-eabe-4b74-99c9-9fb568c71331)



5) 추가 사항 (모델2.py)


InceptionResNetV2를 불러와서 좋은 결과를 얻긴 했지만, 이전에는 ResNet50을 직접 구현해보고 이를 변형하여 정확도를 얻고 싶었다. 따라서, ResNet50을 구현한 후에 해당 모델의 layer를 수정하고 dropout과 regularzation을 추가해본 코드도 올려놓았다. 높은 정확도를 기대했지만 원하는 정확도가 나오지 않아 사용하지 못했었다. 허나, 이를 직접 구현해보고 변형하며 모델에 대한 많은 공부와 학습의 문제점에 대해 알 수 있어서 올리게 되었다.


