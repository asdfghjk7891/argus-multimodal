# LANL AENS(Attack Edge Novelty Score) 계산.
# AENS = 훈련 구간에 없던 공격 엣지 수 / 공격 엣지 수 (고유 (src, dst) 쌍 기준)
# 입력: split_lanl.py 결과 (ts, src, dst, ..., label)
# 실행: python analysis/lanl_aens.py
import os

DST = './data/lanl/'
FILE_DELTA = 10000
DATE_OF_EVIL = 150885   # 훈련 0 ~ 150,885초
END = 1270000           # 테스트 150,885 ~ 1,270,000초

train_edges = set()
att_pairs = set()

c = 0
while c <= END:
    p = DST + f'{c}.txt'
    if os.path.exists(p):
        with open(p) as f:
            for line in f:
                t = line.split(',')
                ts = int(t[0])
                if ts >= END:
                    break
                s, d = int(t[1]), int(t[2])
                if s == d:
                    continue
                if ts < DATE_OF_EVIL:
                    train_edges.add((s, d))
                elif t[-1].strip() == '1':
                    att_pairs.add((s, d))
    c += FILE_DELTA

new = sum(1 for e in att_pairs if e not in train_edges)
print(f'훈련 구간 엣지 수 : {len(train_edges):,}')
print(f'공격 엣지 수      : {len(att_pairs):,}')
print(f'훈련에 없던 공격 엣지 : {new:,}')
print(f'AENS = {new / max(len(att_pairs), 1):.4f}')
