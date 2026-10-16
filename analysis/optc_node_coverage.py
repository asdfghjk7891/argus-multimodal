# OpTC 노드 커버리지 계산 (요건 1).
# 원본 OpTC 그래프(ARGUS) 노드 중 bro conn 플로우에서도 관찰되는 호스트 비율
# 입력: data/nmap.pkl (원본 노드 목록), split_optc_bro.py 결과 (data/optc_bro/flows/*.txt, nmap_bro.pkl)
# 실행: python analysis/optc_node_coverage.py
import glob
import pickle

DATA = './data/'
OLD_NMAP = DATA + 'nmap.pkl'
BRO_DST = DATA + 'optc_bro/'


def to_host(x):
    s = str(x).strip()
    if s.isdigit():
        return f'SysClient{int(s):04d}.systemia.com'
    return s


def pct(a, b):
    return f'{a:,} / {b:,} = {a / max(b, 1):.4f}'


with open(OLD_NMAP, 'rb') as f:
    old = pickle.load(f)
orig_nodes = {to_host(x) for x in old if x is not None}

with open(BRO_DST + 'nmap_bro.pkl', 'rb') as f:
    bro_names = pickle.load(f)

flow_hosts = set()
for p in sorted(glob.glob(BRO_DST + 'flows/*.txt')):
    with open(p) as f:
        for line in f:
            t = line.split(',')
            if len(t) < 3:
                continue
            for i in (1, 2):
                try:
                    flow_hosts.add(bro_names[int(t[i])])
                except (ValueError, IndexError):
                    pass

print(f'원본 그래프 노드 수 : {len(orig_nodes)}')
print(f'노드 커버리지       : {pct(len(orig_nodes & flow_hosts), len(orig_nodes))}')
