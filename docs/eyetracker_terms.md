# Eyetracker 용어 정리

---

## 1. 약어 정리

| 약어 | 풀네임 | 설명 |
|------|--------|------|
| **CLAHE** | Contrast Limited Adaptive Histogram Equalization | 대비 제한 적응형 히스토그램 평활화 — 조명 불균일 보정 필터 |
| **MPIIGaze** | Max Planck Institute for Informatics Gaze | 막스 플랑크 연구소에서 공개한 시선 추정 데이터셋 |
| **PDDL** | Public Domain Dedication and Licence | ODC(Open Data Commons)의 공개 도메인 라이선스 |
| **RGB** | Red, Green, Blue | 일반적인 색상 채널 순서 |
| **BGR** | Blue, Green, Red | OpenCV 기본 색상 채널 순서 (R↔B 반전) |
| **HDF5** | Hierarchical Data Format version 5 | 대용량 수치 데이터를 계층 구조로 저장하는 파일 포맷 |
| **CHW** | Channel, Height, Width | PyTorch 텐서 차원 순서 |
| **HWC** | Height, Width, Channel | OpenCV/NumPy 이미지 차원 순서 |
| **PnP** | Perspective-n-Point | 3D-2D 대응점으로 카메라 자세를 추정하는 알고리즘 |
| **ImageNet** | — | 대규모 이미지 분류 데이터셋. 여기서는 정규화 평균/표준편차 값의 출처로 사용 |
| **LReLU** | Leaky Rectified Linear Unit | 음수 입력에 작은 기울기(0.01)를 허용하는 활성화 함수. Dying ReLU 방지 및 Kaiming Uniform 초기화와 호환 |
| **BN** | Batch Normalization | 미니배치 단위로 각 채널 출력을 정규화하는 레이어. $\text{BN}(x)=\gamma\cdot\frac{x-\mu_B}{\sqrt{\sigma_B^2+\epsilon}}+\beta$. 학습 가능한 scale($\gamma$)·shift($\beta$) 파라미터 각 $C$개 → 총 $2C$개. 학습 안정화·내부 공변량 이동(Internal Covariate Shift) 완화. ResNet에서 Conv `bias=False`와 함께 사용 ($\beta$가 bias 역할 대체). |
| **FC** | Fully Connected (layer) | 모든 입력 뉴런과 출력 뉴런이 연결된 선형 레이어 ($W\mathbf{x}+\mathbf{b}$) |

---

## 2. 수식 개념 정리

### 3. 사용 데이터

| 수식 | 의미 |
|------|------|
| $F = \{f_1, \ldots, f_N\}$ | 파일명 집합, 총 N개 |
| $\texttt{img\_path}_i$ | i번째 샘플의 원본 이미지 경로 (피험자/세션/파일명 조합) |
| $\mathbf{g}_i^L,\, \mathbf{g}_i^R \in \mathbb{R}^3$ | i번째 샘플의 좌/우안 시선 벡터 (3차원) |
| $\|\mathbf{g}\|_2 = 1$ | 시선 벡터는 단위벡터 (크기 = 1) |
| $\bar{\mathbf{g}}_i$ | 좌/우안 시선 벡터의 평균 |
| $\mathbf{g}_i^{\text{label}}$ | 평균 벡터를 다시 정규화한 최종 레이블 단위벡터 |

### 4. 전처리

| 수식 | 의미 |
|------|------|
| $I_i \in \mathbb{R}^{H \times W \times 3}$ | i번째 원본 입력 이미지 (480×640, BGR) |
| $\bar{b}_i$ | 이미지 전체 픽셀의 평균 밝기 ($30 \leq \bar{b}_i \leq 220$ 조건) |
| $\text{sharpness}_i$ | 라플라시안 분산으로 측정한 선명도 ($\geq 50$ 조건) |
| $\theta_i$ | 시선 벡터와 카메라 정면 사이의 각도 ($\leq 40°$ 조건) |
| $\mathcal{L}$ | MediaPipe로 검출한 478개 랜드마크 좌표 집합 (정규화 좌표 $[0,1]^2$) |
| $\mathcal{S}_L,\, \mathcal{S}_R$ | 좌/우안 영역에 해당하는 랜드마크 인덱스 집합 (각 16점) |
| $\text{bbox}$ | 눈 영역 바운딩 박스 (패딩 $p=0.3$ 적용, x1·y1·x2·y2) |
| $\mathbf{p}_{2D},\, \mathbf{p}_{3D}$ | PnP에 사용하는 6개 특징점의 2D 픽셀 / 3D 실세계 좌표 |
| $K_{\text{approx}}$ | 근사 카메라 내부 파라미터 행렬 (초점거리 = 이미지 너비 W) |
| $[\mathbf{R} \mid \mathbf{t}]$ | PnP로 추정한 회전행렬·이동벡터 (카메라 외부 파라미터) |
| $\mathbf{h}_i = [\text{pitch, yaw, roll}]^\top$ | 회전행렬에서 변환한 오일러 각 (단위: rad) |
| $\hat{C}_L \in \mathbb{R}^{3\times64\times64}$ | 최종 저장되는 왼눈 크롭 (64×64, CHW, uint8) |

### 5. Dataset 증강·정규화

| 수식 | 의미 |
|------|------|
| $b \sim \text{Bernoulli}(0.5)$ | 50% 확률로 좌우 반전 여부 결정 |
| $\texttt{flip}_W(I)$ | 이미지를 가로 방향으로 뒤집는 연산 |
| $g_x \to -g_x$ | 좌우 반전 시 시선 벡터의 x성분 부호 반전 |
| $\mathcal{T}$ | albumentations 증강 변환 파이프라인 |
| $\boldsymbol{\mu},\, \boldsymbol{\sigma}$ | ImageNet BGR 채널별 평균·표준편차 |
| $\hat{I}_c = \frac{I_c/255 - \mu_c}{\sigma_c}$ | 채널별 정규화 (결과 범위 $\approx [-2.1,\, 2.6]$) |

### 6. 모델

| 수식 | 의미 |
|------|------|
| $\hat{\mathbf{G}} = f_\theta(\mathbf{L}, \mathbf{R}, \mathbf{H})$ | 모델이 예측한 시선 벡터 (좌눈·우눈·헤드포즈 입력) |
| $\mathbf{G} \in \mathbb{R}^{B\times3},\; \|\mathbf{G}_{i,:}\|_2=1$ | 정답 레이블 단위벡터 (손실 계산에만 사용) |

### 7. 모델 아키텍처

| 수식 | 의미 |
|------|------|
| $\mathbf{f} = \frac{\text{Enc}(L)+\text{Enc}(R)}{2}$ | 좌/우안 인코더 출력의 평균 — KernelNet 특징 벡터 |
| $\mathbf{K} = \text{reshape}(W_2\,\text{LReLU}(W_1\mathbf{f}+\mathbf{b}_1)+\mathbf{b}_2,\ B,3,5,5)$ | KernelNet이 생성하는 동적 컨볼루션 커널 |
| $y = \text{DynamicFilter}(x, \mathbf{K})$ | 샘플별 커널로 채널별 독립 conv 적용 (depthwise) |
| $\mathbf{x}_{in} = [\mathbf{f}_L;\mathbf{f}_R;\mathbf{H}]$ | Regressor 입력 — 좌/우안 특징 + 헤드포즈 concat |
| $\hat{\mathbf{G}} = \mathbf{o}/\|\mathbf{o}\|_2$ | Regressor 출력 정규화 → 단위벡터 강제 |
| Kaiming Uniform ($a=\sqrt{5}$) | LeakyReLU와 호환되는 가중치 초기화 방법 [3] |
