# 학습 가능한 적응형 전처리 필터를 활용한 시선 추정 성능 향상

---

## 1. 실험 목적

- 적응형 커널을 이용한 이미지 필터링이 CNN을 이용한 학습 성능을 향상시킬 수 있는가?
- 조명/외형 변화에 강건한 눈 이미지 전처리를 **학습으로 대체**할 수 있는가를 검증
- 캘리브레이션에 비해 외부 작업 필요도 낮음

기존 동적 필터 연구 [11]가 영상 생성·예측 태스크에 동적 커널을 적용한 것과 달리, 본 연구는 동적 커널을 **시선 인식 태스크의 전처리 단계**에 배치하여 조명·외형 변화에 강건한 특징 추출을 목표로 한다.

### 기대효과

① **시선 추정 정확도 향상**
고정 필터(CLAHE 등) 대비 샘플별 최적 전처리로 angular error 감소 기대

② **조명·외형 변화에 대한 강건성**
in-the-wild 환경의 조명 불균일, 피험자별 외형 차이에 자동 적응

③ **Regressor 경량화**
적응형 전처리가 입력 품질을 높여 Regressor 은닉층을 256으로 축소하면서도 파라미터 효율을 높일 것으로 기대 (Baseline 대비 은닉층 크기 74% 감소: 1024 → 256).
단, 2×2 Ablation 결과 KernelNet+hidden=256(1.4049°)은 KernelNet 없는 hidden=1024(1.3587°)보다 낮은 성능을 보였다. 동일 hidden_dim 조건에서는 KernelNet이 일관되게 우세하나, 은닉층 축소에 따른 성능 손실을 완전히 보상하지는 못한다.

④ **수동 전처리 튜닝 불필요**
환경별 CLAHE 파라미터 등을 수동 조정하지 않아도 됨

---

## 2. 데이터셋 명세

### 2.1 원본 데이터셋 (MPIIGaze) 개요

| 항목 | 내용 |
|------|------|
| 출처 | Max Planck Institute for Informatics |
| 라이선스 | ODC Public Domain Dedication and Licence (PDDL) |
| 피험자 수 | 15명 ($p_{00} \sim p_{14}$) |
| 총 이미지 | 213,659장 |
| 수집 기간 | 피험자당 평균 45.7일 |
| 수집 환경 | 일상 노트북 사용 중 (in-the-wild) |
| 수집 방식 | 10분마다 랜덤 20개 온스크린 위치 응시 (1 세션) |
| 카메라 | 노트북 내장 전면 단안 RGB 카메라 |

> **MPIIGaze 좌표계** : $x$ = 오른쪽, $y$ = 아래, $z$ = 카메라 반대 방향

### 2.2 원본 데이터 구조

```
data/raw/Data/
├── Original/
│     └── p00/
│           └── day01/
│                 ├── 0001.jpg   ← 원본 얼굴 이미지 (가변 해상도)
│                 └── ...
└── Normalized/
      └── p00/
            └── day01.mat       ← gaze + head_pose 레이블
```

### 2.3 .mat 파일 구조

```
mat
├── filenames  (N,)        string
└── data
    ├── left
    │   ├── gaze   (N, 3)      float64  정규화 공간 시선 벡터
    │   ├── image  (N, 36, 60) uint8    정규화 눈 이미지 (grayscale)
    │   └── pose   (N, 2)      float64  정규화 공간 두상 각도
    └── right
        ├── gaze   (N, 3)      float64
        ├── image  (N, 36, 60) uint8
        └── pose   (N, 2)      float64
```

### 2.4 필드 활용 여부

| 필드 | 활용 | 이유 |
|------|:----:|------|
| `filenames` | O | Original/ 이미지 경로 매핑 |
| `left.gaze` | O | 시선 레이블 |
| `right.gaze` | O | 시선 레이블 |
| `left.image` | X | 추론 시 동일 정규화 변환 재현 불가 |
| `right.image` | X | 추론 시 동일 정규화 변환 재현 불가 |
| `left.pose` | X | 정규화 공간 기준 — 추론 시 재현 불가 |
| `right.pose` | X | 정규화 공간 기준 — 추론 시 재현 불가 |

---

## 3. 사용 데이터 명세

### 3.1 원본 데이터셋 내부 필드

**File Names**

$$F = \{f_1, f_2, \ldots, f_N\}, \quad f_i \in \texttt{string}$$

$$\texttt{img\_path}_i = \texttt{Original/} \underbrace{p_k}_{\text{피험자}} / \underbrace{d_j}_{\text{세션}} / f_i$$

**Left.gaze / Right.gaze**

$$G_L,\, G_R \in \mathbb{R}^{N \times 3}, \quad \texttt{dtype: float64}$$

$$\mathbf{g}_i^L,\, \mathbf{g}_i^R \in \mathbb{R}^3, \quad \|\mathbf{g}_i^L\|_2 = \|\mathbf{g}_i^R\|_2 = 1$$

> 정규화 공간(Normalized Space): MPIIGaze 논문 [5]의
> 가상 카메라 변환 공간. $g_z < 0 \Leftrightarrow$ 화면 방향을 향하는 시선.

**레이블 생성 — 좌/우안 평균 → 단위벡터**

$$\bar{\mathbf{g}}_i = \frac{\mathbf{g}_i^L + \mathbf{g}_i^R}{2}$$

$$\mathbf{g}_i^{\text{label}} = \frac{\bar{\mathbf{g}}_i}{\|\bar{\mathbf{g}}_i\|_2} \in \mathbb{R}^3$$

---

## 4. 전처리 명세 (`preprocess.py`)

### 입력

$$I_i \in \mathbb{R}^{H \times W \times 3}, \quad \texttt{dtype: uint8 BGR}$$

> MPIIGaze 원본 이미지는 피험자·기기마다 해상도가 다르다. $H,\,W$는 `img.shape[:2]`로 동적으로 결정된다.

$$\mathbf{g}_i^{\text{label}} \in \mathbb{R}^3 \quad \text{(단위벡터)}$$

### Step 1 — 품질 필터

$$\bar{b}_i = \frac{1}{HW}\sum_{h,w} \text{gray}(I_i)_{h,w}, \qquad 1 \leq \bar{b}_i \leq 220$$

> MPIIGaze 원본 프레임은 어두운 배경을 포함하므로 전체 프레임 평균 밝기가 매우 낮다. 이를 반영하여 하한을 1로 설정한다.

$$\text{sharpness}_i = \text{Var}\!\left(\nabla^2\,\text{gray}(I_i)\right) \geq 2.0$$

$$\theta_i = \arccos\!\left(\text{clip}(-g_z^{\text{label}},\,-1,\,1)\right) \leq 40°$$

세 조건 모두 만족할 때만 이후 단계 진행. 탈락 샘플은 HDF5에 포함되지 않음.

### Step 2 — 눈 Bbox 검출

$$I_{\text{det}} = \texttt{apply\_det}(I_i,\;\texttt{cfg})$$

MediaPipe FaceLandmarker (Tasks API 0.10) 로 478개 랜드마크 검출 [2]:

$$\mathcal{L} = \{(x_k^{\text{norm}},\;y_k^{\text{norm}})\}_{k=0}^{477} \subset [0,1]^2$$

눈 영역 인덱스 $\mathcal{S}_L,\,\mathcal{S}_R \subset \{0,\ldots,477\}$ (각 16점), 패딩 $p = 0.3$:

$$\Delta x = x_2 - x_1, \quad \Delta y = y_2 - y_1$$

$$\text{bbox} = \bigl(\max(0,\,x_1 - p\Delta x),\;\max(0,\,y_1 - p\Delta y),\;\min(W,\,x_2 + p\Delta x),\;\min(H,\,y_2 + p\Delta y)\bigr)$$

$$\text{bbox}_L,\;\text{bbox}_R \in \mathbb{Z}^4 \quad \text{(x1, y1, x2, y2)}$$

### Step 3 — 헤드포즈 추정 (PnP)

$$I_{\text{pose}} = \texttt{apply\_pose}(I_i,\;\texttt{cfg})$$

6개 특징점 (코끝·턱·눈 바깥·입 끝):

$$\mathbf{p}_{2D} \in \mathbb{R}^{6\times2}, \qquad \mathbf{p}_{3D} \in \mathbb{R}^{6\times3} \quad \text{(고정 3D 모델, mm)}$$

근사 내부 파라미터 행렬 ($f_{\text{approx}} = W$):

$$K_{\text{approx}} = \begin{bmatrix} W & 0 & W/2 \\ 0 & W & H/2 \\ 0 & 0 & 1 \end{bmatrix}$$

PnP 솔버 (ITERATIVE):

$$[\mathbf{R} \mid \mathbf{t}] = \texttt{solvePnP}(\mathbf{p}_{3D},\;\mathbf{p}_{2D},\;K_{\text{approx}},\;\mathbf{0})$$

회전행렬 → 오일러 각:

$$\text{sy} = \sqrt{R_{00}^2 + R_{10}^2}$$

$$\mathbf{h}_i = \begin{bmatrix} \text{pitch} \\ \text{yaw} \\ \text{roll} \end{bmatrix} = \begin{bmatrix} \arctan2(R_{21},\; R_{22}) \\ \arctan2(-R_{20},\; \text{sy}) \\ \arctan2(R_{10},\; R_{00}) \end{bmatrix} \in \mathbb{R}^3 \quad [\text{rad}]$$

### Step 4 — 눈 크롭

원본 프레임에서 크롭 (필터 미적용):

$$C_L = I_i[y_1:y_2,\; x_1:x_2] \in \mathbb{R}^{(y_2-y_1)\times(x_2-x_1)\times 3}$$

$$\widetilde{C}_L = \texttt{resize}(C_L,\;64\times64) \in \mathbb{R}^{64\times64\times3} \quad \text{(HWC)}$$

$$\hat{C}_L = \widetilde{C}_L^{\top_{(2,0,1)}} \in \mathbb{R}^{3\times64\times64} \quad \text{(CHW, uint8)}$$

### Step 5 — HDF5 저장

| 필드 | 형상 | dtype | 내용 |
|------|------|-------|------|
| `left_eye` | $(N,3,64,64)$ | uint8 | 왼눈 크롭 CHW BGR |
| `right_eye` | $(N,3,64,64)$ | uint8 | 오른눈 크롭 CHW BGR |
| `gaze` | $(N,3)$ | float32 | 단위벡터 $\mathbf{g}_i^{\text{label}}$ |
| `head_pose` | $(N,3)$ | float32 | $[\text{pitch, yaw, roll}]$ rad |

분할 비율: train $80\%$ / val $10\%$ / test $10\%$ (seed = 42)

---

## 5. Dataset 명세 (`dataset.py` — MPIIGazeDataset)

### 입력 (HDF5)

$$\texttt{left\_eye}_i,\; \texttt{right\_eye}_i \in \mathbb{R}^{3 \times 64 \times 64}, \quad \texttt{dtype: uint8}$$

$$\mathbf{g}_i^{\text{label}} \in \mathbb{R}^3, \qquad \mathbf{h}_i \in \mathbb{R}^3 \quad \text{(head\_pose)}$$

### Step 1 — HorizontalFlip (학습 시 50%)

베르누이 확률변수 $b \sim \text{Bernoulli}(0.5)$에 대해:

$$b = 1 \;\Rightarrow\; \begin{cases} I_L' = \texttt{flip}_W(I_R), \quad I_R' = \texttt{flip}_W(I_L) & \text{(좌우 swap + 픽셀 반전)} \\ g_x^{\text{label}} \leftarrow -g_x^{\text{label}} & \text{($x$성분 부호 반전)} \end{cases}$$

$$\texttt{flip}_W(I)_{:,\,h,\,w} = I_{:,\,h,\,W-1-w}$$

> MPIIGaze 좌표계에서 $x$ = 오른쪽 방향이므로 좌우 반전 시 $g_x \to -g_x$

### Step 2 — albumentations 증강 (학습 시)

$$I \xrightarrow{\mathcal{T}} I' = \mathcal{T}(I)$$

| 변환 | 확률 | 파라미터 |
|------|------|----------|
| `Rotate` | 50% | $\theta \sim \mathcal{U}(-10°, +10°)$ |
| `RandomBrightnessContrast` | 50% | $\pm 0.2$ |
| `GaussNoise` | 30% | — |
| `CoarseDropout` | 20% | 소규모 마스킹 |

#### Rotate 적용 시 헤드포즈 roll 보정

눈 크롭 이미지를 $\theta$도 회전하면, 이미지 기준 좌표계에서 머리가 반대 방향으로 기울어 보이므로 헤드포즈의 roll 성분을 동일한 각도만큼 차감한다.

$$\theta_{\text{rad}} = \theta \times \frac{\pi}{180}$$

$$h_{\text{roll}} \leftarrow h_{\text{roll}} - \theta_{\text{rad}}$$

> albumentations `Rotate`의 양수 $\theta$는 반시계(CCW) 방향 회전이다. 이미지가 CCW로 $\theta$만큼 돌아가면 roll이 $-\theta$만큼 보정된다.
> `Rotate`가 적용되지 않은 샘플($b_{\text{rot}}=0$)은 보정 없음.

### Step 3 — 정규화 (ImageNet BGR 통계)

백본 네트워크로 ImageNet [1] 데이터셋으로 사전 학습된 ResNet18 [4] 아키텍처를 채용하였다. 사전 학습된 가중치가 기억하는 원본 데이터의 통계적 분포를 보존하기 위해, 입력 이미지는 모델에 입력되기 전 ImageNet 채널별 평균 및 표준편차로 정규화되었다. OpenCV BGR 채널 순서에 맞춰 RGB 기준값의 채널 순서를 반전하여 적용하였다.

$$\boldsymbol{\mu} = [0.406,\; 0.456,\; 0.485]^\top, \qquad \boldsymbol{\sigma} = [0.225,\; 0.224,\; 0.229]^\top \quad \text{(BGR 채널 순서)}$$

$$\hat{I}_c = \frac{I_c / 255.0 - \mu_c}{\sigma_c}, \quad c \in \{B, G, R\}$$

$$\hat{I} \in \mathbb{R}^{3 \times 64 \times 64}, \quad \texttt{dtype: float32}, \quad \text{range} \approx [-2.1,\; 2.6]$$

### 출력 (배치 $B$개)

$$\mathbf{L},\; \mathbf{R} \in \mathbb{R}^{B\times3\times64\times64}, \qquad \mathbf{G},\; \mathbf{H} \in \mathbb{R}^{B\times3}$$

---

## 6. 모델 입력 벡터

| 변수 | 형상 | dtype | 출처 | 역할 |
|------|------|-------|------|------|
| $\mathbf{L}$ | $B\times3\times64\times64$ | float32 | `dataset._normalize` | 모델 입력 |
| $\mathbf{R}$ | $B\times3\times64\times64$ | float32 | `dataset._normalize` | 모델 입력 |
| $\mathbf{H}$ | $B\times3$ | float32 | `preprocess.solve_head_pose` | 모델 입력 |
| $\mathbf{G}$ | $B\times3$ | float32 | `.mat` 레이블 | 손실 계산용 |

모델 호출:

$$\hat{\mathbf{G}} = f_\theta(\mathbf{L},\; \mathbf{R},\; \mathbf{H}) \in \mathbb{R}^{B\times3}$$

레이블 (손실 계산용, 모델 입력 아님):

$$\mathbf{G} \in \mathbb{R}^{B\times3}, \quad \|\mathbf{G}_{i,:}\|_2 = 1$$

---

## 7. 모델 아키텍처 명세

### 7.1 SiameseBackbone 커스터마이징

**역할:** 좌/우안 이미지에서 시선 추정에 유용한 시각적 특징 벡터를 추출한다. 동일 가중치를 공유하여 좌/우안 특징이 동일한 표현 공간에 놓이도록 보장한다. Chen & He (2021) [10]은 공유 가중치를 가진 Siamese 구조가 의미적으로 유사한 입력에 대해 일관된 표현을 자연스럽게 강제함을 보였다. 좌/우안은 동일한 기관의 다른 인스턴스이므로 이 특성이 직접 적용된다.

ResNet18 [4] (ImageNet [1] pretrained)에서 마지막 분류 FC 레이어를 제거하고 GlobalAvgPool2d까지만 사용한다.

```python
encoder = nn.Sequential(*list(resnet18.children())[:-1])
# 입력: (B, 3, 64, 64) → 출력: (B, 512)
```

- `freeze_backbone = False` → pretrained 가중치에서 fine-tuning
- 좌/우안에 **동일 가중치(shared weights)** 적용

---

### 7.2 KernelNet

**역할:** 좌/우안 이미지의 조명·질감 특성을 분석하여, 해당 샘플에 최적화된 전처리 필터 커널을 동적으로 생성한다 [11]. 고정 필터(CLAHE 등)와 달리 입력마다 다른 커널을 출력하므로 다양한 촬영 환경에 적응할 수 있다.

**입력:** $L,\ R \in \mathbb{R}^{B \times 3 \times 64 \times 64}$

**출력:** $\mathbf{K} \in \mathbb{R}^{B \times 3 \times 5 \times 5}$

#### Encoder

$L$, $R$ 각각 독립적으로 동일한 Encoder를 통과 (가중치 공유):

$$\mathbf{x}_1^{(\cdot)} = \text{MaxPool}\!\left(\text{LReLU}\!\left(\text{BN}(W_{c1} * (\cdot) + b_{c1})\right)\right) \in \mathbb{R}^{B \times 32 \times 32 \times 32}$$

$$\mathbf{x}_2^{(\cdot)} = \text{MaxPool}\!\left(\text{LReLU}\!\left(\text{BN}(W_{c2} * \mathbf{x}_1^{(\cdot)} + b_{c2})\right)\right) \in \mathbb{R}^{B \times 64 \times 16 \times 16}$$

$$\mathbf{x}_3^{(\cdot)} = \text{AvgPool}\!\left(\text{LReLU}\!\left(\text{BN}(W_{c3} * \mathbf{x}_2^{(\cdot)} + b_{c3})\right)\right) \in \mathbb{R}^{B \times 128}$$

$$\text{where } (\cdot) \in \{L,\; R\}$$

#### Feature 평균

$$\mathbf{f} = \frac{\mathbf{x}_3^{(L)} + \mathbf{x}_3^{(R)}}{2} \in \mathbb{R}^{B \times 128}$$

#### FC → 커널 생성

$$\mathbf{h} = \text{LReLU}(W_1 \mathbf{f} + \mathbf{b}_1) \in \mathbb{R}^{B \times 512}$$

$$\mathbf{k} = W_2 \mathbf{h} + \mathbf{b}_2 \in \mathbb{R}^{B \times 75}$$

$$\mathbf{K} = \text{reshape}(\mathbf{k},\ B,\ 3,\ 5,\ 5) \in \mathbb{R}^{B \times 3 \times 5 \times 5}$$

#### 초기화 전략

FC2의 bias를 5×5 Gaussian 커널($\sigma=1.0$)로 설정하고, weight는 매우 작은 값($\text{std}=0.01$)으로 초기화한다.
학습 초기에는 bias가 지배적이므로 KernelNet 출력이 약한 평활화 필터에 근사하고,
학습이 진행될수록 weight가 커지며 입력에 적응하는 방향으로 커널이 발전한다 [12].

$$
K_{\text{init}} \approx \frac{1}{2\pi\sigma^2} \exp\!\left(-\frac{x^2+y^2}{2\sigma^2}\right), \quad \sigma = 1.0
$$

#### 레이어 명세

| 레이어 | 연산 | 입력 shape | 출력 shape | 파라미터 수 |
|--------|------|-----------|-----------|------------|
| Conv1 | $W_{c1} * x + b_{c1}$, BN, LReLU, MaxPool(2) | (B,3,64,64) | (B,32,32,32) | 896 |
| Conv2 | $W_{c2} * x + b_{c2}$, BN, LReLU, MaxPool(2) | (B,32,32,32) | (B,64,16,16) | 18,496 |
| Conv3 | $W_{c3} * x + b_{c3}$, BN, LReLU, AvgPool | (B,64,16,16) | (B,128) | 73,856 |
| FC1 | $W_1 \mathbf{f} + \mathbf{b}_1$, LReLU | (B,128) | (B,512) | 66,048 |
| FC2 | $W_2 \mathbf{h} + \mathbf{b}_2$ | (B,512) | (B,75) | 38,475 |
| **합계** | | | | **197,771** |

> LReLU = LeakyReLU(negative\_slope=0.01)

**가중치 초기화:** Kaiming Uniform ($a=\sqrt{5}$, PyTorch 기본값) — LeakyReLU와 호환 [3]

---

### 7.3 DynamicFilter

**역할:** KernelNet이 생성한 커널 $\mathbf{K}$를 눈 이미지에 실제로 적용한다. 채널별 독립 합성곱(depthwise convolution)으로 샘플마다 다른 필터링을 수행하며, 공간 해상도는 유지한다.

**입력:** $x \in \mathbb{R}^{B \times 3 \times 64 \times 64}$, $\mathbf{K} \in \mathbb{R}^{B \times 3 \times 5 \times 5}$

**출력:** $y \in \mathbb{R}^{B \times 3 \times 64 \times 64}$

$$x_{\text{flat}} = \text{reshape}(x,\ 1,\ B \cdot C,\ H,\ W) \in \mathbb{R}^{1 \times BC \times H \times W}$$

$$k_{\text{flat}} = \text{reshape}(\mathbf{K},\ B \cdot C,\ 1,\ k,\ k) \in \mathbb{R}^{BC \times 1 \times k \times k}$$

$$y_{\text{flat}} = \text{conv2d}(x_{\text{flat}},\ k_{\text{flat}},\ \text{padding}=\lfloor k/2 \rfloor,\ \text{groups}=B \cdot C)$$

$$y = \text{reshape}(y_{\text{flat}},\ B,\ C,\ H,\ W) \in \mathbb{R}^{B \times 3 \times 64 \times 64}$$

> 동일 커널 $\mathbf{K}$가 $L$, $R$ 양쪽에 적용됨 (좌우 눈 공유 커널)

---

### 7.4 Regressor

**역할:** 좌/우안 특징 벡터와 헤드포즈를 하나의 벡터로 결합한 뒤, FC 레이어를 통해 3차원 시선 방향 벡터를 회귀한다. 최종 출력은 L2 정규화를 통해 단위벡터로 변환되며, 이는 Cosine Similarity 손실 함수의 입력 조건을 만족한다.

**입력:** $\mathbf{f}_L,\ \mathbf{f}_R \in \mathbb{R}^{B \times 512}$, $\mathbf{H} \in \mathbb{R}^{B \times 3}$

**출력:** $\hat{\mathbf{G}} \in \mathbb{R}^{B \times 3}$ (단위벡터)

#### 입력 연결

$$\mathbf{x}_{in} = [\mathbf{f}_L\ ;\ \mathbf{f}_R\ ;\ \mathbf{H}] \in \mathbb{R}^{B \times 1027}$$

#### 레이어 계산

$$\mathbf{h} = \text{Dropout}\!\left(\text{LReLU}(W_1 \mathbf{x}_{in} + \mathbf{b}_1),\ p{=}0.3\right) \in \mathbb{R}^{B \times d_h}$$

$$\mathbf{o} = W_2 \mathbf{h} + \mathbf{b}_2 \in \mathbb{R}^{B \times 3}$$

$$\hat{\mathbf{G}} = \frac{\mathbf{o}}{\|\mathbf{o}\|_2} \in \mathbb{R}^{B \times 3}$$

#### 레이어 명세

| 모델 | 레이어 | 연산 | 입력 shape | 출력 shape | 파라미터 수 |
|------|--------|------|-----------|-----------|------------|
| Proposed | FC1 | $W_1 \mathbf{x}_{in} + \mathbf{b}_1$, LReLU, Dropout | (B,1027) | (B,256) | 263,424 |
| Proposed | FC2 | $W_2 \mathbf{h} + \mathbf{b}_2$ | (B,256) | (B,3) | 771 |
| Proposed | normalize | $\mathbf{o}/\|\mathbf{o}\|_2$ | (B,3) | (B,3) | 0 |
| Proposed | **합계** | | | | **264,195** |
| Baseline | FC1 | $W_1 \mathbf{x}_{in} + \mathbf{b}_1$, LReLU, Dropout | (B,1027) | (B,1024) | 1,052,672 |
| Baseline | FC2 | $W_2 \mathbf{h} + \mathbf{b}_2$ | (B,1024) | (B,3) | 3,075 |
| Baseline | normalize | $\mathbf{o}/\|\mathbf{o}\|_2$ | (B,3) | (B,3) | 0 |
| Baseline | **합계** | | | | **1,055,747** |

**가중치 초기화:** Kaiming Uniform ($a=\sqrt{5}$, PyTorch 기본값) — LeakyReLU와 호환 [3]

---

### 7.5 가중치 초기화 요약

| 모듈 | 초기화 방법 | 근거 |
|------|-----------|------|
| KernelNet Conv/FC1 | Kaiming Uniform ($a=\sqrt{5}$) | PyTorch 기본, LeakyReLU 호환 [3] |
| KernelNet FC2 bias | 5×5 Gaussian ($\sigma=1.0$) | 초기 출력이 약한 평활화 필터에 근사 [12] |
| KernelNet FC2 weight | std=0.01 (거의 0) | 초기엔 bias 지배적, 학습 중 점진적 적응 |
| Regressor FC | Kaiming Uniform ($a=\sqrt{5}$) | PyTorch 기본, LeakyReLU 호환 [3] |
| BatchNorm | weight=1, bias=0 | PyTorch 기본 |
| SiameseBackbone | ImageNet pretrained | fine-tuning (freeze=False) |

---

### 7.6 아키텍처 조합 이유

| 모듈 | 이유 |
|------|------|
| **Siamese** | 좌/우안에 동일 가중치 적용 → 일관된 특징 추출, 파라미터 절감 [10] |
| **KernelNet** | 고정 필터(CLAHE 등) 대신 입력마다 최적 커널을 학습으로 생성 |
| **DynamicFilter** | KernelNet이 생성한 커널을 이미지에 실제 적용하는 연산 레이어 |
| **HeadPose $\mathbf{H}$** | 눈 이미지만으로는 시선의 절대 방향 기준 부족 → 고개 방향으로 보정 |
| **F.normalize** | 손실 함수(Cosine Similarity)가 단위벡터를 가정하므로 출력 정규화 필수 |

---

### 7.7 전체 파라미터 요약

#### 계산 공식

레이어 유형별 파라미터 수:

$$|\theta_{\text{Conv}}| = C_{in} \times C_{out} \times k^2 + C_{out}$$

$$|\theta_{\text{FC}}| = d_{in} \times d_{out} + d_{out}$$

$$|\theta_{\text{BN}}| = 2C \quad \text{(weight + bias)}$$

#### KernelNet 계산

| 레이어 | 계산식 | 파라미터 수 |
|--------|--------|------------|
| Conv1 | $3 \times 32 \times 3^2 + 32 = 864 + 32$ | $896$ |
| Conv2 | $32 \times 64 \times 3^2 + 64 = 18{,}432 + 64$ | $18{,}496$ |
| Conv3 | $64 \times 128 \times 3^2 + 128 = 73{,}728 + 128$ | $73{,}856$ |
| FC1 | $128 \times 512 + 512 = 65{,}536 + 512$ | $66{,}048$ |
| FC2 | $512 \times 75 + 75 = 38{,}400 + 75$ | $38{,}475$ |
| **합계** | $896 + 18{,}496 + 73{,}856 + 66{,}048 + 38{,}475$ | $\mathbf{197{,}771}$ |

#### Regressor 계산

| 레이어 | 계산식 | Proposed | Baseline |
|--------|--------|:--------:|:--------:|
| FC1 | $1027 \times d_h + d_h$ | $1027 \times 256 + 256 = 263{,}424$ | $1027 \times 1024 + 1024 = 1{,}052{,}672$ |
| FC2 | $d_h \times 3 + 3$ | $256 \times 3 + 3 = 771$ | $1024 \times 3 + 3 = 3{,}075$ |
| **합계** | | $\mathbf{264{,}195}$ | $\mathbf{1{,}055{,}747}$ |

#### SiameseBackbone (ResNet18, FC 제거)

$$|\theta_{\text{Backbone}}| = |\theta_{\text{ResNet18}}| - |\theta_{\text{FC}}^{\text{cls}}| = 11{,}689{,}512 - (512 \times 1000 + 1000) = 11{,}689{,}512 - 513{,}000 = 11{,}176{,}512$$

주요 레이어 분해:

| 블록 | 계산식 | 파라미터 수 |
|------|--------|------------|
| conv1 (7×7, 3→64) | $3 \times 64 \times 7^2$ | $9{,}408$ |
| layer1 (64→64, ×2 block) | $4 \times (64 \times 64 \times 3^2) + \text{BN}$ | $\approx 148{,}000$ |
| layer2 (64→128, ×2 block) | $2 \times (64{\times}128{\times}9 + 128{\times}128{\times}9) + \text{downsample}$ | $\approx 526{,}000$ |
| layer3 (128→256, ×2 block) | $2 \times (128{\times}256{\times}9 + 256{\times}256{\times}9) + \text{downsample}$ | $\approx 2{,}100{,}000$ |
| layer4 (256→512, ×2 block) | $2 \times (256{\times}512{\times}9 + 512{\times}512{\times}9) + \text{downsample}$ | $\approx 8{,}393{,}000$ |
| **합계** | | $\approx \mathbf{11{,}176{,}512}$ |

> ResNet의 Conv는 bias=False (BN이 bias 역할 대체), BN만 bias 포함.

#### 전체 합산

**ProposedModel:**

$$|\theta^{\text{Proposed}}| = |\theta_{\text{Backbone}}| + |\theta_{\text{KernelNet}}| + |\theta_{\text{Regressor}}^{\text{P}}|$$

$$= 11{,}176{,}512 + 197{,}771 + 264{,}195 = 11{,}638{,}478 \approx 11.64\text{M}$$

**BaselineModel** (KernelNet 없음):

$$|\theta^{\text{Baseline}}| = |\theta_{\text{Backbone}}| + |\theta_{\text{Regressor}}^{\text{B}}|$$

$$= 11{,}176{,}512 + 1{,}055{,}747 = 12{,}232{,}259 \approx 12.23\text{M}$$

| 모듈 | Proposed | Baseline |
|------|:--------:|:--------:|
| SiameseBackbone (ResNet18, shared) | 11,176,512 | 11,176,512 |
| KernelNet | 197,771 | — |
| Regressor | 264,195 | 1,055,747 |
| **합계** | **11,638,478** | **12,232,259** |

> SiameseBackbone은 좌/우안에 가중치를 공유하므로 1회만 산정.

---

## 8. 학습 알고리즘 명세

### 8.1 손실 함수 — Cosine Loss

$$
\mathcal{L}_{\cos}(\hat{\mathbf{G}}, \mathbf{G}) = \frac{1}{B}\sum_{i=1}^{B}\Bigl(1 - \hat{\mathbf{G}}_i \cdot \mathbf{G}_i\Bigr)
$$

$\hat{\mathbf{G}}_i,\, \mathbf{G}_i \in \mathbb{R}^3$는 단위벡터이므로 내적 = cosine similarity.

| 값 | 의미 |
|----|------|
| 0 | 예측 = 정답 (완전 일치) |
| 1 | 예측 ⊥ 정답 (직교, 90°) |
| 2 | 예측 = −정답 (정반대, 180°) |

**Cosine Loss 채택 이유 — 시선 벡터의 기하학적 특성 활용**

① **Angular error 직접 최소화**

$$
\mathcal{L}_{\cos} = 1 - \cos\theta \quad \Longrightarrow \quad \frac{d\mathcal{L}_{\cos}}{d\theta} = \sin\theta \geq 0 \quad (\theta \in [0°, 180°])
$$

단조 증가 함수이므로 loss 최소화 = angular error 직접 최소화.

② **pitch/yaw 변환 불필요**

$$
\text{pitch} = \arcsin(-g_y), \quad \text{yaw} = \arctan2(g_x,\, -g_z)
$$

arcsin은 $g_y \to \pm1$ 에서 포화(극점 불안정), arctan2는 $\pm\pi$ 불연속(yaw 주기성 문제).
3D 단위벡터를 직접 사용하면 이 변환이 불필요하다.

③ **손실 함수 비교**

| 손실 | 입력 형태 | 문제점 |
|------|-----------|--------|
| MSE(pitch, yaw) | 2D 각도 | 극점 불안정, yaw 주기성, 정보 손실 |
| MSE(3D vector) | 3D 벡터 | angular error와 비선형 관계 |
| **Cosine Loss** | **3D 단위벡터** | **angular error 단조 최소화, 변환 불필요** |

---

### 8.2 옵티마이저 — AdamW

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t
$$
$$
v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2
$$
$$
\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t}
$$
$$
\theta_t \leftarrow \theta_{t-1} - \eta\!\left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\varepsilon} + \lambda\,\theta_{t-1}\right)
$$

마지막 항 $\lambda\,\theta_{t-1}$이 **decoupled weight decay** — gradient update와 분리하여 $\theta$에 직접 적용.

| 하이퍼파라미터 | 값 | 설명 |
|--------------|-----|------|
| $\eta$ | $10^{-4}$ | 기본 학습률 |
| $\beta_1$ | 0.9 | 1차 모멘텀 계수 |
| $\beta_2$ | 0.999 | 2차 모멘텀 계수 |
| $\varepsilon$ | $10^{-8}$ | 수치 안정항 |
| $\lambda$ | $10^{-4}$ | weight decay 계수 |

**AdamW 채택 이유**

Alkhalid (2022) [6]는 MNIST Siamese 네트워크에서 Adam이 SGD(93%)·Adadelta(82%) 대비 97% 정확도를 달성함을 실험적으로 확인했다. 단, Adam+L2 정규화는 적응형 학습률 때문에 파라미터마다 effective weight decay가 달라지는 문제가 있다 [7]. 본 모델은 pretrained Backbone [4](작은 gradient)과 random-init KernelNet(큰 gradient)이 공존하므로 이 불균일성이 실질적이다. AdamW는 weight decay를 gradient와 분리하여 $\theta$에 직접 적용함으로써 이 문제를 해결한다 [7].

---

### 8.3 LR 스케줄러

**기본: CosineAnnealingLR**

$$
\eta_e = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\frac{\pi\, e}{E}\right)
$$

$e$: 현재 epoch, $E$: 총 epoch 수, $\eta_{\min} = 0$.

CosineAnnealingLR은 SGDR [8]에서 제안된 cosine 감쇠 방식을 warm restart 없이 적용한 것이다.

**warmup 사용 시: SequentialLR (LinearLR → CosineAnnealingLR)**

$$
\eta_e = \begin{cases}
\eta_{\max} \cdot \dfrac{e}{e_w} & 0 \leq e < e_w \\[6pt]
\eta_{\min} + \dfrac{1}{2}(\eta_{\max}-\eta_{\min})\!\left(1+\cos\dfrac{\pi(e-e_w)}{E-e_w}\right) & e_w \leq e \leq E
\end{cases}
$$

$e_w$: warmup epoch 수. warmup은 pretrained Backbone의 fine-tuning 초기 불안정을 방지하기 위해 사용한다 [9].

---

### 8.4 Gradient Clipping

$$
\mathbf{g} \leftarrow \mathbf{g} \cdot \min\!\left(1,\; \frac{\delta}{\|\mathbf{g}\|_2}\right), \quad \delta = 5.0
$$

$\mathbf{g}$: 전체 파라미터의 gradient를 이어붙인 벡터. $\|\mathbf{g}\|_2 > 5.0$ 이면 크기를 5.0으로 정규화한다. KernelNet이 생성하는 동적 커널의 불안정 gradient를 억제한다.

---

### 8.5 학습 모드

**E2E (End-to-End)**

모든 파라미터(Backbone + KernelNet + Regressor)를 처음부터 함께 학습한다.

$$
\theta^* = \arg\min_{\theta_{\text{all}}}\; \mathcal{L}_{\cos}
$$

**Sequential**

| 단계 | epoch 범위 | 학습 파라미터 | 동결 파라미터 |
|------|-----------|--------------|-------------|
| Phase 1 | $[1,\; r \cdot E]$ | Regressor | Backbone + KernelNet |
| Phase 2 | $[r \cdot E + 1,\; E]$ | 전체 | — |

$$
r \in (0,1): \text{freeze ratio}, \quad E: \text{총 epoch 수}
$$

| 모드 | 장점 | 단점 |
|------|------|------|
| E2E | 구현 단순, 전체 동시 최적화 | Backbone 초기 불안정 가능 |
| Sequential | Backbone 안정 보존 | 하이퍼파라미터($r$) 추가 |

---

### 8.6 평가 지표

**샘플별 angular error**

$$
\epsilon_i = \arccos\!\Bigl(\mathrm{clip}(\hat{\mathbf{G}}_i \cdot \mathbf{G}_i,\;-1,\;1)\Bigr) \times \frac{180°}{\pi}
$$

clip은 부동소수점 오차로 인한 $|\hat{\mathbf{G}} \cdot \mathbf{G}| > 1$ 방지.

**MAE (Mean Angular Error)**

$$
\text{MAE}_{\text{angular}} = \frac{1}{N}\sum_{i=1}^{N} \epsilon_i \quad [\text{degrees}]
$$

**Best model 기준**: validation set의 `val_angular_err` 최소 epoch에서 `best.pt` 저장.
`val_loss`(Cosine Loss) 기준이 아님에 유의.

---

## 참고문헌

[1] Deng, J., Dong, W., Socher, R., Li, L. J., Li, K., & Fei-Fei, L. (2009).
ImageNet: A large-scale hierarchical image database.
In *2009 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)* (pp. 248–255).

[2] Google. (n.d.). *MediaPipe Face Landmarker*. https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker

[3] He, K., Zhang, X., Ren, S., & Sun, J. (2015).
Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification.
In *Proceedings of the IEEE International Conference on Computer Vision (ICCV)* (pp. 1026–1034).

[4] He, K., Zhang, X., Ren, S., & Sun, J. (2016).
Deep residual learning for image recognition.
In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)* (pp. 770–778).

[5] Zhang, X., Sugano, Y., Fritz, M., & Bulling, A. (2015).
Appearance-based gaze estimation in the wild.
In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)* (pp. 4511–4520).

[6] Alkhalid, F. F. (2022).
The effect of optimizers on Siamese Neural Network performance.
In *Proceedings of the International Conference on Industrial Engineering and Operations Management (IEOM)*, Istanbul, Turkey, March 7–10, 2022 (pp. 5084–5089).

[7] Loshchilov, I., & Hutter, F. (2019).
Decoupled weight decay regularization.
In *Proceedings of the International Conference on Learning Representations (ICLR)*.

[8] Loshchilov, I., & Hutter, F. (2017).
SGDR: Stochastic gradient descent with warm restarts.
In *Proceedings of the International Conference on Learning Representations (ICLR)*.

[9] He, T., Zhang, Z., Zhang, H., Zhang, Z., Xie, J., & Li, M. (2019).
Bag of tricks for image classification with convolutional neural networks.
In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)* (pp. 558–567).

[10] Chen, X., & He, K. (2021).
Exploring simple Siamese representation learning.
In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)* (pp. 15750–15758).

[11] De Brabandere, B., Jia, X., Tuytelaars, T., & Van Gool, L. (2016).
Dynamic filter networks.
In *Advances in Neural Information Processing Systems (NeurIPS)*.

[12] Tek, F. B. (2020).
Adaptive convolution kernel for artificial neural networks.
arXiv:2009.06385.
