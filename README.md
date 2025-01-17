# **UWB 센서 객체 위치 분류**

---

## **1. 프로젝트 개요**  
- **주제:** UWB 센서를 이용한 vit 모델 기반 위치 분류  
- **목적:** 비구속적 센서를 활용하여 객체의 위치를 확인하고 추후 취약 계층의 낙상사고와 같은 비정상적인 움직임 감지를 목표로 함. 
- **사용 데이터:** 2개의 UWB 레이더를 이용하여 움직임 시나리오에 맞게 측정을 진행.

---

## **2. 데이터 구성**  

### **2.1. 데이터 수집**  
- **장비:** UWB radar 2개  
- **샘플링 주파수:** 17 Hz (1초당 17 프레임 수집)  
- **실험 대상:** 60명(학습:48명)
- **수집 방법:**
     ![image](https://github.com/user-attachments/assets/3626c5ec-9d77-47b6-bc39-9372c90efcd6)
   - 나누어진 구역을 3초에 한번씩 이동하는 시나리오(5분)
- **데이터 크기:**  
   - 피험자 당 **4284 프레임** 

### **2.2. 데이터 형태**      
- 4개의 채널별 **360(거리) X 4284(시간)** 크기의 행렬    
---

## **3. 데이터 전처리**  

### **3.1. 데이터 정규화**   
- **정규화 기법:** Z-score Normalization을 거리 축에 적용  
   -  $z = \frac{x - \mu}{\sigma}$  

### **3.2. threshold 설정과 LOG 적용**  
- $threshold = \mu + 3\sigma$
- 값들에 LOG를 적용하여 신호가 더 도드라지게 처리.

### **3.3. moving everage filter**
- 거리 인덱스가 세밀하게 나누어져 있기에 움직이는 경우 비어있는 인덱스 값이 존재
- moving everage filter를 넓게 적용하여 움직임이 넓게 퍼지도록 함.

### **3.4. 데이터 형태 변경**  
- **기존 데이터 형태:** 360 X 4284
- 360은 거리, 4284는 시간 축에 해당함.
- 5분동안 움직임 구역을 3초마다 움직이고 14개 구역을 6번 왕복하는 데이터를 얻을 수 있었음.
- 4284 = 17(fps) X 3(sec) X 14(전체 구역 수) X 6(주기)
- **reshape:** 360 x 51 x 84
- 360 x 51은 3초마다 객체의 위치정보를 지닌 데이터고 피험자당 84개의 데이터를 얻음.  
- **비정형성 감소:**
- 360 x 51 데이터는 거리축이 큼. 구역을 정해둔 실험 시나리오이기 때문에 분류 성능의 증가를 위해 5개씩 합하는 방식으로 거리축을 조정함.
- **최종** 72 x 51 x 84

---

## **4. 모델링**  
- 모델은 레이더 각각 하나씩 사용(com,tv로 명명)
### **4.1. CNN 모델**  
- **입력 데이터:** (batch, 1, 51, 72)  
- **구조:**
   - **Convolution block**  
        - **Conv2D Layer**: 시간축과 거리축을 기준으로 특징 추출
        - **BatchNormalization**: batch간 normalization을 적용하여 피험자간의 생길 수 있는 차이를 보완
        - **Relu**: activation function
        - **Pooling Layer**: Max Pooling으로 다운샘플링
   - **Flatten**
   - **Dense block**: 최종 출력층에서 레이더 각각 9개,7개를 각각 분류  
- **성과:** 테스트 정확도 com: 약 **79.64%**, tv: 약 **84.32%**

### **4.2. CNN lightening 모델**  
- **입력 데이터:** (batch, 1, 51, 72)    
- **구조:**  
   - CNN 모델과 같은 구조 사용
   - lightening 프레임워크 사용 및 5-fold cross validation 적용
   - 훈련 데이터와 테스트 데이터 분할시 랜덤하게 분할하기 위한 **GroupKfold** 방식 적용
- **성과:** 테스트 정확도 com: 약 **85.11%**, tv: 약 **94.71%**

### **4.3. Vision Transformer 모델**  
- **데이터 준비:** pretrain dataset의 특성에 맞게 기존 데이터를 RGB 형태로 변환  
- **모델:** Pretrained Vision Transformer (ViT) Fine-tuning
     - huggingface 라이브러리를 이용하여 cifar10으로 pretrained ViT를 사용
     - vit_fold_model_selection.py 파일을 이용하여 데이터셋에 맞는 모델을 탐색 후 사용.
  
- **최종 모델**: vit_small_patch16_224
---

## **5. 프로젝트 결과 요약**  
- **최종 성능**  
   ![image](https://github.com/user-attachments/assets/89eba9ee-50b4-4fba-af5b-bf2de39ff620)
 
---

## **6. 고찰**
1. **Preprocessing**
- 레이더마다 모델을 다르게 사용하였고 라벨링도 다르게 하였는데 이는 상호 보완적인 결과가 더 좋을 것으로 예상했기 때문.
- 동일한 모델로 두 데이터를 학습시켜보는 시도가 필요해보임.

2. **Experiment**
- 실험 시나리오를 기역자 구역을 왕복하도록 구성하였는데 이는 너무 사용 범위가 제한적임.
- 구역을 좀 더 확장하고 다양한 이동 시나리오를 구상해보았으면 더 높은 확장성을 지니지 않았을까 생각함. 
- **Vital-sign monitoring and spatial tracking of multiple people using a contactless radar-based sensor**
- 위 논문에 따르면 피험자가 멈춰 있는 경우에 호흡과 같은 생체 데이터를 추출해낼 수 있음.
- 3초마다 움직이는 시나리오라면 3초동안 멈춰있는 실험 데이터를 통한 예측 호흡 수를 추출해보았으면 좋았을 것 같음.
