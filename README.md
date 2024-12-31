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
- **실험 대상:** 60명 
- **수집 방법:**
   -  ![image](https://github.com/user-attachments/assets/3ea50f4b-0f1d-4faa-8548-b14c94936006)
   - 나누어진 구역을 3초에 한번씩 이동하는 시나리오(5분)
- **데이터 크기:**  
   - 피험자 당 **4284 프레임** 

### **2.2. 데이터 저장 및 파일 구조**  
- 데이터 파일: **label_data_7500_96_{subject_id}.mat**  
- 구조:  
   - 4개의 채널별 **7500x24** 크기의 행렬  
   - 주파수 성분으로 24개의 주파수 대역 데이터를 포함  

---

## **3. 데이터 전처리**  

### **3.1. 데이터 병합 및 정규화**  
- **4채널 데이터 병합:** 4개의 PVDF 센서 데이터를 하나로 병합  
- **정규화 기법:** Z-score Normalization  
   -  \( z = \frac{x - \mu}{\sigma} \)  
   - 채널 간 신호 강도의 차이를 최소화  

### **3.2. 시간-주파수 변환**  
- **방법:** Continuous Wavelet Transform (CWT)  
- **대상 주파수 대역:** 1~5 Hz  
- **결과 데이터 크기:** (7500x24) → 시간축(7500)과 주파수축(24)  

### **3.3. 데이터 분할**  
- **윈도우 사이즈:** 7500 프레임 (30초)  
- **Step Size:** 1250 프레임 (5초 간격)  
- **샘플 수:** 전체 데이터를 5초 단위로 분할 → 여러 샘플 생성  

---

## **4. 모델링**  

### **4.1. 2D CNN 모델**  
- **입력 데이터:** (4, 7500, 24)  
- **구조:**  
   - **Conv2D Layer**: 시간축과 주파수축을 기준으로 특징 추출  
   - **Pooling Layer**: Max Pooling으로 다운샘플링  
   - **Fully Connected Layer**: 최종 출력층에서 4가지 자세 분류  
- **성과:** 테스트 정확도 약 **79.17%**

### **4.2. 3D CNN 모델**  
- **입력 데이터:** (1, 7500, 24, 4)  
   - 시간, 주파수, 채널 간의 상호작용을 학습  
- **구조:**  
   - **Conv3D Layer**: 시간, 주파수, 채널을 동시에 학습  
   - **Pooling Layer**: Max Pooling 적용  
   - **Fully Connected Layer**: 자세 분류  
- **성과:** 테스트 정확도 **86.37%**  

### **4.3. Vision Transformer 모델**  
- **데이터 준비:** CWT 변환 데이터를 이미지화하여 저장  
- **모델:** Pretrained Vision Transformer (ViT) Fine-tuning  

---

## **5. 프로젝트 결과 요약**  
- **최종 성능 비교:**  
   | 모델          | 정확도 (%) | Precision | Recall | F1-score |  
   |---------------|------------|-----------|--------|----------|  
   | **2D CNN**    | 79.17      | 0.80      | 0.79   | 0.79     |  
   | **3D CNN**    | 86.37      | 0.88      | 0.86   | 0.87     |  

---

## **6. 추가 개선 사항**
1. ** Preprocessing**
- pvdf_processing.m
 Zcore normalization을 진행한 후 데이터를 저장할 때 7500 프레임을 기준으로 데이터를 걸렀는데 이후 데이터 샘플링을 할 때, step size를 설정이 제한적이었음. 이 부분을 수정해서 다시 전처리 하면 좋겠음.
- DownSampling
현재 1초에 250프레임임. 해상도가 너무 높아 정보의 의미가 모호한 것으로 판단됨. 또한 데이터 용량이 너무 커져 모든 데이터를 학습시키지 못했음.

- 호흡대역을 제외한 데이터셋과 호흡대역을 포함한 데이터셋 두 세트로 나누어 학습

2. **Experiment**
- 실험을 다시 진행할 수 있다면 pvdf 채널을 좀 더 퍼뜨려놓을 수 있었으면 좋겠음. 이미지화 했을 때, 자세가 좀 더 명확하게 나타날 것이라고 생각됨.

