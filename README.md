# MAGNET: Multi-modal Attack-aware Graph NETwork

인증 로그와 네트워크 플로우를 모달리티별 인코더로 따로 처리한 뒤 임베딩 단계에서 결합하는(Late Fusion) 그래프 기반 내부 이동 탐지 모델.
[ARGUS](https://github.com/C0ldstudy/Argus) (Xu et al., IEEE S&P 2024) 코드를 기반으로 구현하였다.

인천대학교 정보통신공학과 졸업논문 코드 (202302916 이용비)

```
인증 특징 (3차원)  -> Auth Encoder  (GCN x3 + NNConv) -> z_auth  --+
                                                                  +-> 결합 층 -> GRU -> 내적 디코더 -> 이상 점수
플로우 특징 (7차원) -> Flows Encoder (GCN x3 + NNConv) -> z_flows --+

z_fused = tanh(W_f · [z_auth || z_flows] + b_f)
```

## 구조

```
main.py                  실행 (옵션, 데이터셋/모델 선택)
classification.py        학습, 컷오프 결정, 테스트, 결과 저장
utils.py                 평가 지표
models/
  argus.py               인코더 (Uni-Auth, Uni-Flows, Early Fusion, Late Fusion)와 손실 함수
  recurrent.py           GRU / LSTM
loaders/
  split_*.py             원본 데이터 전처리
  load_*.py              스냅샷 그래프 로더
  tdata.py, load_utils.py
analysis/
  lanl_coverage.py       LANL 노드 커버리지 (요건 1)
  optc_node_coverage.py  OpTC 노드 커버리지 (요건 1)
  cert_node_coverage.py  CERT 노드 커버리지 (요건 1)
  lanl_aens.py           LANL AENS (요건 4)
  optc_aens.py           OpTC AENS (요건 4)
libauc/                  ARGUS에 포함된 LibAUC (AP loss, SOAP optimizer)
```

## 환경

- Windows 10, Python 3.9, PyTorch 2.0.1 (CUDA 11.8), PyTorch Geometric 2.2.0
- Intel i5-10400F, RAM 32GB, NVIDIA RTX 3060Ti (8GB)

```
conda create -n argus python==3.9
conda activate argus
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv torch_geometric==2.2.0 --find-links https://data.pyg.org/whl/torch-2.0.1+cu118.html
pip install -r requirements.txt
```

## 데이터 준비

모든 명령은 저장소 최상위 폴더에서 실행한다. 전처리 결과는 `./data/` 아래에 저장되며, 원본 위치가 다르면 각 `split_*.py` 상단의 경로를 바꾼다.

**LANL** — https://csr.lanl.gov/data/cyber1/ 에서 `auth.txt`, `flows.txt`, `redteam.txt`를 받아 `data/`에 둔다.
```
python loaders/split_lanl.py
```

**OpTC** — ARGUS가 배포한 OpTC 전처리 데이터(`data/*.txt`, `data/nmap.pkl`, `data/redteam_optc.txt`)와 DARPA OpTC의 ecar-bro, bro conn 로그가 필요하다. ecar-bro와 ARGUS 데이터 사이의 시간 오프셋(8,028초) 보정이 포함되어 있다.
```
python loaders/split_optc_bro.py
```

**CERT r6.2** — https://kilthub.cmu.edu/articles/dataset/Insider_Threat_Test_Dataset/12841247 에서 `r6.2`, `answers`를 받는다. http.csv가 커서 전처리에 2시간 이상 걸린다.
```
python loaders/split_cert.py
```

**PicoDomain** — https://github.com/iHeartGraph/PicoDomain 에서 `Zeek_Logs`, `Red Log.xlsx`를 받는다.
```
python loaders/split_picodomain.py
```

## 실행

공통 설정: 학습률 0.01, patience 3, BCE loss, seed 0~4.
데이터가 `./data/`에 없으면 `--data_path`로 지정한다 (LANL은 `data\lanl\`, 나머지는 `data\`).

### LANL

```
# Uni-Auth
python main.py --dataset LANL --delta 1 --lr 0.01 --loss bce --gpu --seed 0
# Uni-Flows
python main.py --dataset LANL --delta 1 --lr 0.01 --loss bce --gpu --seed 0 --flows --fusion uniflows
# Early Fusion (ARGUS 원래 구성)
python main.py --dataset LANL --delta 1 --lr 0.01 --loss bce --gpu --seed 0 --flows --fusion early
# Late Fusion (MAGNET)
python main.py --dataset LANL --delta 1 --lr 0.01 --loss bce --gpu --seed 0 --flows --fusion late
```

가설 검증 실험은 Late Fusion 명령에 옵션을 추가한다.

| 옵션 | 실험 |
|---|---|
| `--coverage 0.3` | H1: 플로우 레코드 30%만 사용 |
| `--gap 500000` | H2: 훈련과 테스트 사이 500,000초 공백 |
| `--auth_stats` | H3: 플로우 특징 대신 인증 통계 사용 |

플로우 특징은 기본적으로 ARGUS 원래 방식(10,000초 파일 단위)으로 집계된다. 스냅샷마다 그 시간의 플로우만 집계하려면 `--flows_per_snapshot`을 추가한다.

### OpTC

bro 플로우가 기록된 구간에 맞춘 설정 (훈련 0~76,000초, 테스트 560,000~690,000초)
```
python main.py --dataset OPTC --bro --delta 1 --lr 0.01 --loss bce --gpu --seed 0
python main.py --dataset OPTC --bro --delta 1 --lr 0.01 --loss bce --gpu --seed 0 --flows --fusion late
```

### CERT r6.2

`--delta`는 일 단위.
```
python main.py --dataset CERT --delta 1 --lr 0.01 --loss bce --gpu --seed 0
python main.py --dataset CERT --delta 1 --lr 0.01 --loss bce --gpu --seed 0 --flows --fusion late
```

### PicoDomain

```
python main.py --dataset PICO --delta 1 --lr 0.01 --loss bce --gpu --seed 0
python main.py --dataset PICO --delta 1 --lr 0.01 --loss bce --gpu --seed 0 --flows --fusion late
```

결과는 `Exps/result/<시각>_seed<N>/result.csv`, `result.txt`에 저장되고 `Exps/all_results.csv`에 누적된다.

## 결과 (LANL, seed 0~4 평균)

| 방법 | 평균 AP | 평균 Recall | ΔAP |
|---|:---:|:---:|:---:|
| Uni-Auth | 0.1397 | 0.6034 | — |
| Uni-Flows | 0.1961 | 0.8050 | +40.4% |
| Early Fusion (ARGUS 원래 구성) | 0.2101 | 0.7198 | +50.4% |
| **Late Fusion (MAGNET)** | **0.2525** | **0.8287** | **+80.7%** |

| 가설 | 조건 | AP | 판정 |
|---|---|:---:|:---:|
| H1 플로우 커버리지 | 플로우 30%만 사용 (seed 0) | 0.3693 → 0.3694 | 기각 |
| H2 훈련/테스트 연속성 | 500,000초 공백 (seed 0) | 0.3693 → 0.4372 | 기각 |
| H3 데이터 이종성 | 플로우를 인증 통계로 대체 (seed 0~4) | 0.2525 → 0.2495 | 기각 |

## ARGUS 대비 변경 사항

- `main.py`: `--flows` 동작 수정(`store_false` → `store_true`), `--seed`, `--data_path`, `--fusion`, `--bro`, H1~H3 옵션 추가, CERT/PicoDomain 추가
- `models/argus.py`: Late Fusion(LANL, OpTC, PicoDomain), Uni-Flows, OpTC Early Fusion, CERT 모델 추가
- `classification.py`: 결과 자동 저장, fusion/seed별 체크포인트
- `loaders/`: OpTC ecar-bro, CERT, PicoDomain 전처리와 로더 추가, LANL H1/H3 설정 추가
- `analysis/`: 데이터셋 요건(노드 커버리지, AENS) 계산

## 참고

- J. Xu, X. Shu, and Z. Li, "Understanding and Bridging the Gap Between Unsupervised Network Representation Learning and Security Analytics," IEEE S&P 2024. https://github.com/C0ldstudy/Argus
- A. D. Kent, "Comprehensive, Multi-Source Cyber-Security Events," Los Alamos National Laboratory, 2015.
- DARPA, Operationally Transparent Cyber (OpTC) Data Release, 2020.
- J. Glasser and B. Lindauer, "Bridging the Gap: A Pragmatic Approach to Generating Insider Threat Data," IEEE S&P Workshops, 2013.
- S. Laprade, B. Bowman, and H. H. Huang, "PicoDomain: A Compact High-Fidelity Cybersecurity Dataset," arXiv:2008.09192, 2020.
