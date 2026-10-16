# CERT r6.2 노드 커버리지 계산 (요건 1).
# 주 그래프(http) 노드 중 보조 모달리티(device)에도 나타나는 노드 비율
# 입력: split_cert.py 결과 (data/cert/<ts>.txt, data/cert/flows/<ts>.txt)
# 실행: python analysis/cert_node_coverage.py
import glob

CERT = './data/cert/'


def read(files):
    nodes = set()
    for p in files:
        with open(p) as f:
            for line in f:
                t = line.split(',')
                if len(t) < 3:
                    continue
                try:
                    nodes.add(int(t[1]))
                    nodes.add(int(t[2]))
                except ValueError:
                    continue
    return nodes


def pct(a, b):
    return f'{a:,} / {b:,} = {a / max(b, 1):.4f}'


main_nodes = read(sorted(glob.glob(CERT + '[0-9]*.txt')))
aux_nodes = read(sorted(glob.glob(CERT + 'flows/[0-9]*.txt')))
print(f'노드 커버리지 : {pct(len(main_nodes & aux_nodes), len(main_nodes))}')
