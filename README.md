# MAGNET: Multi-modal Attack-aware Graph NETwork
**정보통신공학과 202302916 이용비**

기존 ARGUS (Xu et al., IEEE S&P 2024)의 미완성된 멀티모달 구조를 완성하고 발전시키는 졸업논문 연구입니다.

> **MAGNET** (**M**ulti-modal **A**ttack-aware **G**raph **NET**work): 인증 로그와 네트워크 플로우를 Late Fusion으로 결합하여 동적 그래프 기반 비지도 내부자 위협 탐지 성능을 향상시키는 프레임워크

---

## 연구 개요

- **Base Paper**: ARGUS: Understanding and Bridging the Gap Between Unsupervised Network Representation Learning and Security Analytics (IEEE S&P 2024)
- **Original Repository**: https://github.com/C0ldstudy/Argus
- **연구 목표**: 기존 ARGUS의 미완성된 멀티모달 구조(auth 로그 + 네트워크 플로우)를 완성하고, Early Fusion의 불안정성을 Late Fusion 구조로 개선
- **핵심 기여**:
  1. ARGUS Late Fusion 구조 완성 (LANL 데이터셋 AP +80.7%)
  2. OpTC bro 네트워크 센서 전처리 파이프라인 구현 및 타임스탬프 동기화 문제(8,028초) 해결
  3. H1(flows 커버리지), H2(훈련/테스트 연속성), H3(데이터 이종성) 가설 검증 실험
  4. 멀티모달 이상 탐지를 위한 **데이터셋 적합성 요건 6가지** 도출
  5. PicoDomain 데이터셋에서 요건 6/6 충족 시 멀티모달 TP +94% 추가 검증

---

## 환경 설정

### 요구 사항
- OS: Windows 10 (Linux/Mac도 가능)
- GPU: NVIDIA GPU (VRAM 8GB 이상 권장)
- CUDA: 11.8 이상
- Python: 3.9

### 1단계: Anaconda 설치
https://www.anaconda.com/download 에서 설치 후 Anaconda Prompt를 통해 실행하였습니다.

### 2단계: 가상환경 생성 및 활성화
```bash
conda create -n argus python==3.9
conda activate argus
```

### 3단계: PyTorch 설치 (CUDA 11.8 기준)
```bash
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
```

> **참고**: CUDA 버전 확인 방법: `nvidia-smi` 명령어 실행 후 우측 상단 CUDA Version 확인
> CUDA 12.x 환경에서도 CUDA 11.8 기반 PyTorch가 정상 작동합니다 (하위 호환)

### 4단계: PyTorch Geometric 및 의존 패키지 설치
```bash
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv torch_geometric==2.2.0 --find-links https://data.pyg.org/whl/torch-2.0.1+cu118.html
```

### 5단계: 나머지 패키지 설치
```bash
pip install -r requirements.txt
```

### 설치 확인
```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```
`2.0.1+cu118`와 `True`가 출력되면 정상입니다.

---

## 데이터셋 준비

### LANL 데이터셋
1. https://csr.lanl.gov/data/cyber1/ 접속 (신청 후 승인 필요)
2. `auth.txt.gz`, `flows.txt.gz`, `redteam.txt.gz` 다운로드
3. 압축 해제 후 아래 경로에 저장

```
data/
├── auth.txt
├── flows.txt
└── redteam.txt
```

전처리 실행:
```bash
python loaders/split_lanl.py
```

### OpTC 데이터셋
1. DARPA OpTC Dataset 다운로드
2. ecar-bro 및 bro conn 파일 준비

전처리 실행:
```bash
python loaders/split_optc_bro.py
```

> **주의**: OpTC는 타임스탬프 오프셋 8,028초 보정이 전처리 코드에 포함되어 있습니다.

### CERT r6.2 데이터셋
1. https://kilthub.cmu.edu/articles/dataset/Insider_Threat_Test_Dataset/12841247 접속
2. `r6.2.tar.bz2`, `answers.tar.bz2` 다운로드 및 압축 해제

전처리 실행:
```bash
python loaders/split_cert.py
```

> **주의**: http.csv가 약 90GB로 전처리에 2시간 이상 소요됩니다.

### PicoDomain 데이터셋
1. https://github.com/iHeartGraph/PicoDomain 에서 `Zeek_Logs.7z` 다운로드
2. `Red Log.xlsx` 다운로드

전처리 실행:
```bash
python loaders/split_picodomain.py
```

---

## 실행 방법

### 기본 실행 (환경 준비)
```bash
conda activate argus
cd C:\Users\user\Desktop\Argus
```

### LANL 데이터셋

```bash
# 단일모달
python main.py --dataset LANL --delta 1 --lr 0.01 --loss bce --gpu --seed 0 --data_path C:\Users\user\Desktop\Argus\data\lanl\

# Late Fusion 멀티모달 (MAGNET)
python main.py --dataset LANL --delta 1 --lr 0.01 --loss bce --gpu --seed 0 --data_path C:\Users\user\Desktop\Argus\data\lanl\ --flows --fusion late

# Early Fusion
python main.py --dataset LANL --delta 1 --lr 0.01 --loss bce --gpu --seed 0 --data_path C:\Users\user\Desktop\Argus\data\lanl\ --flows --fusion early

# Uni-Flows (flows만 단독)
python main.py --dataset LANL --delta 1 --lr 0.01 --loss bce --gpu --seed 0 --data_path C:\Users\user\Desktop\Argus\data\lanl\ --flows --fusion uniflows
```

### OpTC 데이터셋

```bash
# 단일모달
python main.py --dataset OPTC --delta 1 --lr 0.01 --loss bce --gpu --seed 0

# Late Fusion 멀티모달
python main.py --dataset OPTC --delta 1 --lr 0.01 --loss bce --gpu --seed 0 --flows --fusion late
```

### CERT r6.2 데이터셋

```bash
# 단일모달
python main.py --dataset CERT --delta 1 --lr 0.01 --loss bce --gpu --seed 0 --data_path C:\Users\user\Desktop\Argus\data\

# Late Fusion 멀티모달
python main.py --dataset CERT --delta 1 --lr 0.01 --loss bce --gpu --seed 0 --data_path C:\Users\user\Desktop\Argus\data\ --flows --fusion late
```

### PicoDomain 데이터셋

```bash
# 단일모달
python main.py --dataset PICO --delta 1 --lr 0.01 --loss bce --gpu --seed 0 --data_path C:\Users\user\Desktop\Argus\data\

# Late Fusion 멀티모달
python main.py --dataset PICO --delta 1 --lr 0.01 --loss bce --gpu --seed 0 --data_path C:\Users\user\Desktop\Argus\data\ --flows --fusion late
```

### 주요 옵션 설명

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--dataset` | `LANL` | 데이터셋 선택 (`LANL`, `OPTC`, `CERT`, `PICO`) |
| `--data_path` | `None` | 데이터셋 경로 직접 지정 |
| `--delta` | `1` | 스냅샷 크기 (시간 단위) |
| `--lr` | `0.01` | 학습률 |
| `--loss` | `default` | 손실 함수 (`default`, `ap`, `bce`) |
| `--gpu` | `False` | GPU 사용 여부 |
| `--seed` | `0` | 랜덤 시드 |
| `--flows` | `False` | flows 데이터 사용 여부 |
| `--fusion` | `early` | 멀티모달 융합 방식 (`early`, `late`, `uniflows`) |

### 실험 결과 저장 위치
```
Exps/
├── result/
│   └── 날짜시간_seed번호/
│       ├── result.txt
│       └── result.csv
└── all_results.csv
```

---

## 프레임워크 구조

```
인증 로그 (auth)  → GCN + NNConv(3차원) → auth_emb  ──┐
                                                         ├→ MLP Fusion → GRU → Anomaly Score
네트워크 플로우   → GCN + NNConv(7차원) → flows_emb ──┘
(flows)
```

**Late Fusion 수식:**
```
z_fused = tanh(W_f · [z_auth || z_flows] + b_f)
```

---

## 기존 ARGUS 대비 변경사항

### `main.py`
- `--flows` 옵션 버그 수정 (`store_false` → `store_true`)
- `--data_path` 옵션 추가 (Windows 백슬래시 자동 변환)
- `--seed` 옵션 추가 + `set_seed()` 함수
- `--fusion` 옵션 추가 (`early`, `late`, `uniflows`)
- `CERT`, `PICO` 데이터셋 옵션 추가

### `models/argus.py`
- `Argus_LANL_LateFusion`: Late Fusion 구조 구현
- `Argus_LANL_UniFlows`: Uni-Flows(flows 단독) 구조
- `Argus_OPTC_LateFusion`: OpTC Late Fusion
- `Argus_OPTC_EarlyFusion`: OpTC Early Fusion
- `Argus_CERT`: CERT 멀티모달 구조
- `Argus_PicoDomain_LateFusion`: PicoDomain Late Fusion 구조

### `classification.py`
- `save_results()` 함수 추가 (자동 결과 저장)

### 신규 파일
| 파일 | 설명 |
|------|------|
| `loaders/split_optc_bro.py` | OpTC bro 전처리 (타임스탬프 보정 포함) |
| `loaders/load_optc_bro.py` | OpTC bro 로더 |
| `loaders/split_cert.py` | CERT r6.2 전처리 |
| `loaders/load_cert.py` | CERT r6.2 로더 |
| `loaders/split_picodomain.py` | PicoDomain 전처리 |
| `loaders/load_picodomain.py` | PicoDomain 로더 |

---

## 실험 결과

### 데이터셋 적합성 요건 체크리스트

멀티모달 이상 탐지가 효과적으로 작동하기 위한 6가지 요건:

| # | 요건 | LANL | OpTC | CERT | PicoDomain |
|---|------|:----:|:----:|:----:|:----:|
| ① | flows 커버리지 ≥70% | ✅ | ❌ | ✅ | ✅ |
| ② | 훈련/테스트 연속성 | ✅ | ❌ | ✅ | ✅ |
| ③ | 진정한 데이터 이종성 | ✅ | ⚠️ | ⚠️ | ✅ |
| ④ | 공격이 그래프 엣지에 반영 | ✅ | ⚠️ | ❌ | ✅ |
| ⑤ | 노드 매핑 일관성 | ✅ | ⚠️ | ✅ | ✅ |
| ⑥ | 타임스탬프 동기화 | ✅ | ⚠️ | ✅ | ✅ |
| | **충족 수** | **6/6** | **2/6** | **4/6** | **6/6** |

### LANL 데이터셋 (seed=0~4 평균)

| 방식 | 평균 AP | 평균 Recall | 유효 seed | ΔAP |
|:----:|:----:|:----:|:----:|:----:|
| 단일모달 (ARGUS) | 0.1397 | 0.6034 | 4/5 | — |
| Uni-Flows | 0.1961 | 0.8050 | 5/5 | +40.4% |
| Early Fusion | 0.2101 | 0.7198 | 3/5 | +50.4% |
| **Late Fusion (MAGNET)** | **0.2525** | **0.8287** | **5/5** | **+80.7%** |
| ARGUS 논문 보고값 | 0.3227 | — | — | — |

### 가설 검증 실험 (LANL 기준)

| 가설 | 실험 조건 | 결과 | 판정 |
|------|------|:----:|:----:|
| H1: flows 커버리지 부족 | flows 30% 절삭 | AP 0.3693→0.3694 | 기각 |
| H2: 훈련/테스트 불연속 | 500,000초 gap | AP 0.3693→0.4372 | 기각 |
| H3: 데이터 이종성 부족 | auth 통계로 flows 대체 | AP 0.2525→0.2495 (-1.2%) | 기각 |

→ LANL에서 Late Fusion 효과는 다양한 조건 변화에 **강건(robust)**함 확인

### 전체 데이터셋 비교

| 데이터셋 | 요건 충족 | 단일모달 AP | 멀티모달 AP | ΔAP | ΔTP |
|------|:----:|:----:|:----:|:----:|:----:|
| LANL | 6/6 | 0.1397 | **0.2525** | **+80.7%** ✅ | — |
| PicoDomain | 6/6 | 0.4561 | **0.4582** | +0.5% | **+94%** ✅ |
| CERT r6.2 | 4/6 | 0.0011 | 0.0008 | -27% ❌ | — |
| OpTC | 2/6 | 0.2950 | 0.2807 | -4.9% ❌ | — |

**핵심 발견**: 요건 6/6 충족 데이터셋(LANL, PicoDomain)에서만 멀티모달 효과 확인

---

## 실험 환경
- OS: Windows 10
- CPU: Intel i5-10400F / RAM: 32GB
- GPU: NVIDIA RTX 3060Ti (VRAM 8GB)
- Python 3.9 / PyTorch 2.0.1 (CUDA 11.8)
- conda 환경명: argus

---

## 참고문헌
- Xu, Z., et al. (2024). ARGUS: Understanding and Bridging the Gap Between Unsupervised Network Representation Learning and Security Analytics. *IEEE S&P 2024*.
- Kent, A. (2015). Comprehensive, multi-source cyber-security events. *Los Alamos National Laboratory*.
- DARPA. (2020). OpTC: Operationally Transparent Cyber Dataset.
- Glasser, J., & Lindauer, B. (2013). Bridging the gap: A pragmatic approach to generating insider threat data. *IEEE S&P Workshops 2013*.
- iHeartGraph. (2020). PicoDomain. https://github.com/iHeartGraph/PicoDomain
