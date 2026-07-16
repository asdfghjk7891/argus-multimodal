# ARGUS Multimodal Extension
**정보통신공학과 202302916 이용비**

기존 ARGUS (Xu et al., IEEE S&P 2024)의 미완성된 멀티모달 구조를 완성하고 발전시키는 졸업논문 연구입니다.

---

## 연구 개요

- **Base Paper**: ARGUS: Understanding and Bridging the Gap Between Unsupervised Network Representation Learning and Security Analytics (IEEE S&P 2024)
- **Original Repository**: https://github.com/C0ldstudy/Argus
- **연구 목표**: 기존 ARGUS의 미완성된 멀티모달 구조(auth 로그 + 네트워크 플로우)를 완성하고, Early Fusion의 불안정성을 Late Fusion 구조로 개선

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

### LANL 데이터셋 다운로드
1. https://csr.lanl.gov/data/cyber1/ 접속
2. 아래 3개 파일 다운로드 (신청 후 승인 필요)
   - `auth.txt.gz` (인증 로그, 약 73GB)
   - `flows.txt.gz` (네트워크 플로우, 약 5.2GB)
   - `redteam.txt.gz` (공격 레이블)
3. 압축 해제 후 아래 경로에 저장

```
C:/Users/user/Desktop/Argus/data/
├── auth.txt        ← 압축 해제한 파일
├── flows.txt       ← 압축 해제한 파일
└── redteam.txt     ← 압축 해제한 파일
```

> **주의**: auth.txt는 용량이 매우 크므로 (73GB) 충분한 저장 공간이 필요합니다.

### 데이터 전처리
`loaders/split_lanl.py` 상단의 경로를 본인 환경에 맞게 수정합니다.

```python
# loaders/split_lanl.py 상단 경로 설정 부분
RED = 'C:/Users/user/Desktop/Argus/data/redteam.txt'  # redteam.txt 경로
SRC = 'C:/Users/user/Desktop/Argus/data/auth.txt'     # auth.txt 경로
DST = 'C:/Users/user/Desktop/Argus/data/lanl/'        # 전처리 결과 저장 폴더
SRC_DIR = 'C:/Users/user/Desktop/Argus/data/'         # flows.txt가 있는 폴더
```

전처리 실행 :
```bash
cd C:\Users\user\Desktop\Argus
conda activate argus
python loaders/split_lanl.py
```

전처리 완료 후 `data/lanl/` 폴더 구조:
```
data/lanl/
├── flows/          ← flows 전처리 결과 (txt 파일 다수)
│   ├── 0.txt
│   ├── 10000.txt
│   └── ...
├── nmap.pkl        ← 노드 매핑
├── umap.pkl        ← 유저 매핑
├── pomap.pkl       ← 포트 매핑
├── prmap.pkl       ← 프로토콜 매핑
├── aomap.pkl       ← 인증 방향 매핑
├── atmap.pkl       ← 인증 타입 매핑
├── ltmap.pkl       ← 로그온 타입 매핑
├── smap.pkl        ← 성공/실패 매핑
├── 0.txt           ← auth 전처리 결과 (txt 파일 다수)
├── 10000.txt
└── ...
```

---

## 실행 방법

### 기본 실행 (환경 준비)
```bash
conda activate argus
cd C:\Users\user\Desktop\Argus
```

### 단일모달 실험 (auth 로그만 사용)
```bash
python main.py --dataset LANL --delta 1 --lr 0.01 --loss bce --gpu --seed 0 --data_path C:\Users\user\Desktop\Argus\data\lanl\
```

### Early Fusion 멀티모달 실험 (auth + flows)
```bash
python main.py --dataset LANL --delta 1 --lr 0.01 --loss bce --gpu --seed 0 --data_path C:\Users\user\Desktop\Argus\data\lanl\ --flows
```

### Late Fusion 멀티모달 실험 (본 연구 개선 방식)
```bash
python main.py --dataset LANL --delta 1 --lr 0.01 --loss bce --gpu --seed 0 --data_path C:\Users\user\Desktop\Argus\data\lanl\ --flows --fusion late
```

### 주요 옵션 설명

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--dataset` | `LANL` | 데이터셋 선택 (`LANL` 또는 `OPTC`) |
| `--data_path` | `None` | 데이터셋 경로 직접 지정 |
| `--delta` | `1` | 스냅샷 크기 (시간 단위, 1 = 1시간) |
| `--lr` | `0.01` | 학습률 |
| `--loss` | `default` | 손실 함수 (`default`, `ap`, `bce`) |
| `--gpu` | `False` | GPU 사용 여부 (플래그) |
| `--seed` | `0` | 랜덤 시드 |
| `--flows` | `False` | flows 데이터 사용 여부 (플래그) |
| `--fusion` | `early` | 멀티모달 융합 방식 (`early` 또는 `late`) |

### 실험 결과 저장 위치
```
Exps/
├── result/
│   └── 날짜시간_seed번호/
│       ├── result.txt   ← 개별 실험 결과
│       └── result.csv
└── all_results.csv      ← 전체 실험 누적 결과
```

---

## 기존 ARGUS 대비 변경사항

### 1. `main.py`

#### ① `--flows` 옵션 버그 수정
기존 코드에서 `--flows` 옵션이 `store_false`로 설정되어 있어 flows 데이터가 실질적으로 항상 비활성화되어 있었습니다.

```python
# 기존 (버그)
ap.add_argument('--flows', action='store_false')

# 수정 후
ap.add_argument('--flows', action='store_true')
```

#### ② `--data_path` 옵션 추가
데이터셋 경로를 명령어 옵션으로 지정할 수 있도록 추가했습니다. Windows 백슬래시 자동 변환을 포함합니다.

```python
ap.add_argument('--data_path', type=str, default=None)

if args.data_path is not None:
    args.data_path = args.data_path.replace('\\', '/')
    if not args.data_path.endswith('/'):
        args.data_path += '/'
```

#### ③ `--seed` 옵션 추가 + `set_seed()` 함수 분리
재현성 확보를 위해 랜덤 시드를 명령어 옵션으로 지정할 수 있도록 추가했습니다.

```python
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

ap.add_argument('--seed', type=int, default=0)
```

#### ④ `--fusion` 옵션 추가
멀티모달 융합 방식(Early Fusion / Late Fusion)을 선택할 수 있도록 추가했습니다.

```python
ap.add_argument('--fusion', type=str, default='early', choices=['early', 'late'])
```

---

### 2. `classification.py`

#### `save_results()` 함수 추가
실험 결과를 자동으로 저장하는 함수를 추가했습니다.
- 개별 실험 결과: `Exps/result/날짜시간_seed번호/result.txt`, `result.csv`
- 누적 실험 결과: `Exps/all_results.csv`

---

### 3. `models/argus.py`

#### ① `Argus_LANL`: `ea_dim` 동적 설정
flows 사용 여부에 따라 NNConv 입력 차원을 동적으로 설정합니다.

```python
self.use_flows = data_kws.get('use_flows', False)
ea_dim = 10 if self.use_flows else 3  # flows 없음: 3개 / flows 있음: 10개
nn4 = nn.Sequential(nn.Linear(ea_dim, 8), nn.ReLU(), nn.Linear(8, h_dim * z_dim))
```

#### ② `Argus_LANL_LateFusion` 클래스 추가
Early Fusion의 불안정성을 개선하기 위해 Late Fusion 구조를 새로 구현했습니다.

**Early Fusion (기존):**
```
auth feature (3개) ──┐
                     ├→ NNConv → embedding → GRU → decoder
flows feature (7개) ─┘
```

**Late Fusion (개선):**
```
auth 그래프  → GCN + NNConv(3차원) → auth_emb  ──┐
                                                    ├→ MLP fusion → GRU → decoder
flows 그래프 → GCN + NNConv(7차원) → flows_emb ──┘
```

---

## 실험 결과 요약 (LANL, BCE Loss, seed=0~4)

### 단일모달 vs Early Fusion vs Late Fusion 평균 비교

| 방식 | 평균 AP | 평균 Recall | 평균 TP | 평균 FP | 평균 FN | 유효 seed |
|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| 단일모달 | 0.1397 | 0.6034 | 240.6 | 3,469.4 | 198.4 | 4/5 |
| Early Fusion | 0.2101 | 0.7198 | 312.0 | 3,184.2 | 123.8 | 3/5 |
| **Late Fusion** | **0.2525** | **0.8287** | **362.6** | **3,123.4** | **73.4** | **5/5** |

### seed별 AP 비교

| seed | 단일모달 | Early Fusion | Late Fusion |
|:----:|:----:|:----:|:----:|
| 0 | 0.0976 | 0.2323 | **0.3693** |
| 1 | 0.1519 | 0.1401 | **0.1713** |
| 2 | 0.2434 | 0.2389 | **0.2502** |
| 3 | 0.0602 | 0.1242 | **0.3343** |
| 4 | 0.1452 | **0.3148** | 0.1376 |

### 주요 성과
- Late Fusion이 단일모달 대비 평균 AP **+80.7%** 향상
- Late Fusion이 Early Fusion 대비 평균 AP **+20.2%** 향상
- Late Fusion은 모든 seed에서 Recall **0.77 이상** 안정적 유지
- Late Fusion seed=0에서 ARGUS 논문 원본 결과(AP 0.3227) 초과 달성 (AP **0.3693**)

---

## 실험 환경
- OS: Windows 10
- CPU: Intel i5-10400F / RAM: 32GB
- GPU: NVIDIA RTX 3060Ti (VRAM 8GB)
- Python 3.9 / PyTorch 2.0.1 (CUDA 11.8)
- conda 환경명: argus

---
