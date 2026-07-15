# ARGUS Multimodal Extension
**정보통신공학과 202302916 이용비**

기존 ARGUS (Xu et al., IEEE S&P 2024)의 미완성된 멀티모달 구조를 완성하고 발전시키는 졸업논문 연구입니다.

---

## 연구 개요

- **Base Paper**: ARGUS: Understanding and Bridging the Gap Between Unsupervised Network Representation Learning and Security Analytics (IEEE S&P 2024)
- **Original Repository**: https://github.com/C0ldstudy/Argus
- **연구 목표**: 기존 ARGUS의 미완성된 멀티모달 구조(auth 로그 + 네트워크 플로우)를 완성하고, Early Fusion의 불안정성을 Late Fusion 구조로 개선

---

## 기존 ARGUS 대비 변경사항

### 1. `main.py`

#### ① `--flows` 옵션 수정
기존 코드에서 `--flows` 옵션이 `store_false`로 설정되어 있어 flows 데이터가 실질적으로 항상 비활성화되어 있었습니다.

```python
# 기존 
ap.add_argument('--flows', action='store_false')

# 수정 후
ap.add_argument('--flows', action='store_true')
```

#### ② `--data_path` 옵션 추가
데이터셋 경로를 명령어 옵션으로 지정할 수 있도록 추가했습니다. Windows 백슬래시 자동 변환을 포함합니다.

```python
ap.add_argument('--data_path', type=str, default=None,
                help='데이터셋 경로 지정. 예: --data_path C:/Users/user/Desktop/Argus/data/lanl/')

# 경로 처리 (백슬래시 자동 변환)
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

결과 저장 폴더명에도 seed 번호가 포함됩니다.
```python
OUTPATH = './Exps/result/' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + f'_seed{args.seed}' + '/'
```

#### ④ `--fusion` 옵션 추가
멀티모달 융합 방식(Early Fusion / Late Fusion)을 선택할 수 있도록 추가했습니다.

```python
ap.add_argument('--fusion', type=str, default='early',
                choices=['early', 'late'],
                help='멀티모달 융합 방식: early (기본값) 또는 late')
```

---

### 2. `classification.py`

#### `save_results()` 함수 추가
실험 결과를 자동으로 저장하는 함수를 추가했습니다.

- **개별 실험 결과**: `Exps/result/날짜시간_seed번호/result.txt`, `result.csv`
- **누적 실험 결과**: `Exps/all_results.csv` (전체 실험 통합)

---

### 3. `models/argus.py`

#### ① `Argus_LANL`: `ea_dim` 동적 설정
flows 사용 여부에 따라 NNConv 입력 차원을 동적으로 설정하도록 수정했습니다.

```python
# 기존 
nn4 = nn.Sequential(nn.Linear(3, 8), nn.ReLU(), nn.Linear(8, h_dim * z_dim))

# 수정 후 
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

```python
class Argus_LANL_LateFusion(GCN):
    def __init__(self, ...):
        # Auth Encoder: auth edge feature (3차원) 독립 인코딩
        self.c4_auth = NNConv(h_dim, z_dim, auth_nn, aggr='mean')   # 3차원 입력
        
        # Flows Encoder: flows edge feature (7차원) 독립 인코딩
        self.c4_flows = NNConv(h_dim, z_dim, flows_nn, aggr='mean') # 7차원 입력
        
        # Fusion MLP: concat(auth_emb, flows_emb) → z_dim으로 압축
        self.fusion = nn.Sequential(nn.Linear(z_dim * 2, z_dim), nn.Tanh())

    def forward_once(self, mask_enum, i):
        ea_auth  = ea[:, :3]   # auth feature (3차원)
        ea_flows = ea[:, 3:]   # flows feature (7차원)
        
        auth_emb  = self.c4_auth(x_auth, ei, edge_attr=ea_auth)
        flows_emb = self.c4_flows(x_flows, ei, edge_attr=ea_flows)
        
        fused = torch.cat([auth_emb, flows_emb], dim=1)
        return self.fusion(fused)
```

#### ③ `detector_lanl_late_rref` 함수 추가

```python
def detector_lanl_late_rref(loader, kwargs, h_dim, z_dim, **kws):
    device = kwargs.pop('device')
    return DetectorEncoder(
        Argus_LANL_LateFusion(loader, kwargs, h_dim, z_dim, device),
        device, 'LANL'
    )
```

---

## 실험 결과 요약 (LANL, BCE Loss)

### 단일모달 vs Early Fusion vs Late Fusion (seed=0~4 평균)

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
- Late Fusion은 모든 seed에서 Recall **0.77 이상** 안정적 유지 (Early Fusion은 seed=1에서 0.276으로 급락)
- Late Fusion seed=0에서 ARGUS 논문 원본 결과(AP 0.3227) 초과 달성 (AP **0.3693**)

---

## 실행 방법

### 환경 설정
```bash
conda activate argus
cd C:\Users\user\Desktop\Argus
```

### 단일모달 실험
```bash
python main.py --dataset LANL --delta 1 --lr 0.01 --loss bce --gpu --seed 0 --data_path C:\Users\user\Desktop\Argus\data\lanl\
```

### Early Fusion 멀티모달 실험
```bash
python main.py --dataset LANL --delta 1 --lr 0.01 --loss bce --gpu --seed 0 --data_path C:\Users\user\Desktop\Argus\data\lanl\ --flows
```

### Late Fusion 멀티모달 실험
```bash
python main.py --dataset LANL --delta 1 --lr 0.01 --loss bce --gpu --seed 0 --data_path C:\Users\user\Desktop\Argus\data\lanl\ --flows --fusion late
```

---

## 실험 환경
- OS: Windows 10
- CPU: Intel i5-10400F / RAM: 32GB
- GPU: NVIDIA RTX 3060Ti (VRAM 8GB)
- Python 3.9 / PyTorch 2.0.1 (CUDA 11.8)
- conda 환경명: argus

---

