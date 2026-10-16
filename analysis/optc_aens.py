# OpTC AENS(Attack Edge Novelty Score) 계산.
# AENS = 훈련 구간에 없던 공격 엣지 수 / 공격 엣지 수 (고유 (src, dst) 쌍 기준)
# 원본 OpTC 그래프와 ecar-bro 그래프(split_optc_bro.py 결과)를 각각 계산
# 실행: python analysis/optc_aens.py
import os

DATA = './data/'
BRO = DATA + 'optc_bro/'


def aens(folder, label_col, tr_end, te_start, te_end, name):
    files = [f for f in os.listdir(folder) if f.endswith('.txt') and f[:-4].isdigit()]
    files.sort(key=lambda x: int(x[:-4]))
    train, attack = set(), set()
    for f in files:
        with open(folder + f, encoding='utf-8', errors='replace') as fp:
            for line in fp:
                l = line.strip().split(',')
                if len(l) <= label_col:
                    continue
                try:
                    ts, s, d, y = int(l[0]), int(l[1]), int(l[2]), int(l[label_col])
                except ValueError:
                    continue
                if s == d:
                    continue
                if ts < tr_end:
                    train.add((s, d))
                elif te_start <= ts < te_end and y == 1:
                    attack.add((s, d))
    novel = len(attack - train)
    print(f"[{name}]")
    print(f"  훈련 엣지 {len(train):,} / 공격 엣지 {len(attack):,} / 훈련에 없던 공격 엣지 {novel}")
    print(f"  AENS = {novel}/{len(attack)} = {novel / max(len(attack), 1):.4f}\n")


# 원본 OpTC 그래프 (훈련 0~573,290 / 테스트 573,290~745,983)
aens(DATA, 8, 573290, 573290, 745983, "원본 OpTC 그래프")

# ecar-bro 그래프 (훈련 0~560,000 / 테스트 560,000~)
aens(BRO, 3, 560000, 560000, 10**9, "ecar-bro 그래프")
