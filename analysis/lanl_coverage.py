# LANL 보조 모달리티(플로우) 커버리지 계산 (요건 1).
# 노드 커버리지 NC = |V_auth ∩ V_flows| / |V_auth|
# 엣지 단위 커버리지와 양 끝 노드가 모두 플로우에 있는 인증 엣지 비율도 같이 출력
# 입력: split_lanl.py 결과 (DST/<ts>.txt, DST/flows/<ts>.txt)
# 실행: python analysis/lanl_coverage.py
import os

DST = './data/lanl/'
FILE_DELTA = 10000
END = 1270000
DATE_OF_EVIL = 150885


def files(sub):
    c = 0
    while c < END:
        p = DST + (sub + '/' if sub else '') + f'{c}.txt'
        if os.path.exists(p):
            yield p
        c += FILE_DELTA


def read_edges(sub):
    E = {'train': set(), 'test': set()}
    for p in files(sub):
        with open(p) as f:
            for line in f:
                t = line.split(',')
                ts = int(t[0])
                if ts >= END:
                    break
                s, d = int(t[1]), int(t[2])
                if s == d and sub == '':
                    continue
                E['train' if ts < DATE_OF_EVIL else 'test'].add((s, d))
    return E


def nodes(E):
    n = set()
    for s, d in E:
        n.add(s)
        n.add(d)
    return n


def pct(a, b):
    return f'{a:,} / {b:,} = {a / max(b, 1):.4f}'


E_auth = read_edges('')
E_flow = read_edges('flows')

for part in ('train', 'test', 'all'):
    ea = (E_auth['train'] | E_auth['test']) if part == 'all' else E_auth[part]
    ef = (E_flow['train'] | E_flow['test']) if part == 'all' else E_flow[part]
    na, nf = nodes(ea), nodes(ef)
    both = sum(1 for s, d in ea if s in nf and d in nf)
    print(f'[{part}]')
    print(f'  노드 커버리지                      : {pct(len(na & nf), len(na))}')
    print(f'  양 끝 노드가 플로우에 있는 인증 엣지 : {pct(both, len(ea))}')
    print(f'  엣지 단위 커버리지                 : {pct(len(ea & ef), len(ea))}')
