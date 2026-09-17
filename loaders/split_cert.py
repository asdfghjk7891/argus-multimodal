"""
split_cert.py v6
주 모달리티: http.csv (user→domain 접근) ← 공격 신호 있음
보조 모달리티: device.csv (USB Connect/Disconnect) ← 추가 공격 신호

출력:
  - data/cert/[ts].txt        : ts, src_id, dst_id, label (http 기반)
  - data/cert/flows/[ts].txt  : ts, src_id, dst_id, connect_cnt, disconnect_cnt (device 기반)
  - data/cert/nmap.pkl        : 노드ID 매핑
  - data/cert/meta.pkl        : 메타 정보
"""

import os
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm

# ── 경로 설정 ─────────────────────────────────────────────────────────────────
CERT_DIR    = 'C:/Users/user/Desktop/r6.2/'
ANSWERS_DIR = 'C:/Users/user/Desktop/answers/'
DST_DIR     = 'C:/Users/user/Desktop/Argus/data/cert/'
DELTA       = 86400   # 1일 단위 (http가 많아서 1일이 적절)
FILE_DELTA  = 86400

os.makedirs(DST_DIR, exist_ok=True)
os.makedirs(os.path.join(DST_DIR, 'flows'), exist_ok=True)

# ── Step 1: 공격자 시간 범위 로드 ─────────────────────────────────────────────
print('[Step 1] 공격자 시간 범위 로드...')

ATTACKERS = {'ACM2278', 'CMP2946', 'PLJ1771', 'CDE1846', 'MBG3183'}
attacker_windows = {}

insiders = pd.read_csv(os.path.join(ANSWERS_DIR, 'insiders.csv'))
insiders_r62 = insiders[insiders['dataset'].astype(float) == 6.2]
print(f'  r6.2 공격자 행 수: {len(insiders_r62)}')

for _, row in insiders_r62.iterrows():
    user = str(row['user']).strip()
    try:
        start_ts = int(pd.to_datetime(str(row['start']).strip()).timestamp())
        end_ts   = int(pd.to_datetime(str(row['end']).strip()).timestamp())
        attacker_windows[user] = (start_ts, end_ts)
        print(f'  {user}: {str(row["start"]).strip()} ~ {str(row["end"]).strip()}')
    except Exception as e:
        try:
            start_str = str(row['start']).strip().lstrip('/')
            end_str   = str(row['end']).strip().lstrip('/')
            start_ts  = int(pd.to_datetime(start_str).timestamp())
            end_ts    = int(pd.to_datetime(end_str).timestamp())
            attacker_windows[user] = (start_ts, end_ts)
            print(f'  {user} (보정): {start_str} ~ {end_str}')
        except Exception as e2:
            print(f'  경고: {user} 파싱 실패 ({e2})')

print(f'  공격자 시간 범위 로드: {len(attacker_windows)}명')

# ── Step 2: 노드 ID 매핑 구축 (http 기반) ─────────────────────────────────────
print('\n[Step 2] 노드 ID 매핑 구축...')
# http.csv: user → domain (URL에서 도메인 추출)
# 노드: user + domain
print('  http.csv에서 도메인 추출 중...')
chunksize = 500000
all_users   = set()
all_domains = set()

http_chunks = pd.read_csv(os.path.join(CERT_DIR, 'http.csv'),
                           usecols=['user', 'url'],
                           chunksize=chunksize)
for chunk in tqdm(http_chunks, desc='도메인 추출'):
    chunk['domain'] = chunk['url'].str.extract(r'https?://([^/]+)', expand=False)
    chunk = chunk.dropna(subset=['domain'])
    all_users.update(chunk['user'].unique())
    all_domains.update(chunk['domain'].unique())

users   = sorted(all_users)
domains = sorted(all_domains)

user_to_id   = {u: i for i, u in enumerate(users)}
domain_to_id = {d: i + len(users) for i, d in enumerate(domains)}
nmap         = users + domains

pickle.dump(nmap, open(os.path.join(DST_DIR, 'nmap.pkl'), 'wb'))
print(f'  사용자 수:  {len(users)}')
print(f'  도메인 수:  {len(domains)}')
print(f'  노드 수:    {len(nmap)}')

# ── Step 3: 시작 epoch 결정 ───────────────────────────────────────────────────
print('\n[Step 3] 시작 epoch 결정...')
df_date = pd.read_csv(os.path.join(CERT_DIR, 'http.csv'),
                      usecols=['date'], nrows=1000000)
df_date['ts_dt'] = pd.to_datetime(df_date['date'], format='%m/%d/%Y %H:%M:%S')

# 전체 범위는 logon과 동일 (2010-01-02 ~ 2011-06-01)
start_epoch = int(pd.to_datetime('2010-01-02 02:19:18').timestamp())
end_epoch   = int(pd.to_datetime('2011-06-01 07:26:02').timestamp())

print(f'  시작 epoch: {start_epoch}')
print(f'  총 기간: {(end_epoch - start_epoch) // 86400}일')

if attacker_windows:
    earliest_attack_epoch = min(v[0] for v in attacker_windows.values())
    earliest_attack_rel   = earliest_attack_epoch - start_epoch
    DATE_OF_EVIL = max(0, earliest_attack_rel - DELTA)
else:
    DATE_OF_EVIL = int((end_epoch - start_epoch) * 0.7)

print(f'  DATE_OF_EVIL: {DATE_OF_EVIL}초 ({DATE_OF_EVIL//86400}일)')
print(f'  → 훈련: 0 ~ {DATE_OF_EVIL//86400}일')
print(f'  → 테스트: {DATE_OF_EVIL//86400} ~ {(end_epoch-start_epoch)//86400}일')

# ── Step 4: http.csv → auth 슬라이스 생성 ────────────────────────────────────
print('\n[Step 4] auth 슬라이스 생성 (http 기반)...')

attacker_rel_windows = {
    user: (s - start_epoch, e - start_epoch)
    for user, (s, e) in attacker_windows.items()
}

slice_data    = {}
total_events  = 0
attack_events = 0
attack_slices = set()

http_chunks = pd.read_csv(os.path.join(CERT_DIR, 'http.csv'),
                           usecols=['date', 'user', 'url', 'activity'],
                           chunksize=chunksize)

for chunk in tqdm(http_chunks, desc='http→auth 처리'):
    chunk = chunk.copy()
    chunk['ts_dt']    = pd.to_datetime(chunk['date'], format='%m/%d/%Y %H:%M:%S')
    chunk['ts']       = chunk['ts_dt'].view(np.int64) // 10**9 - start_epoch
    chunk['domain']   = chunk['url'].str.extract(r'https?://([^/]+)', expand=False)
    chunk = chunk.dropna(subset=['domain'])
    chunk['src_id']   = chunk['user'].map(user_to_id).fillna(-1).astype(int)
    chunk['dst_id']   = chunk['domain'].map(domain_to_id).fillna(-1).astype(int)
    chunk = chunk[(chunk['ts'] >= 0) & (chunk['src_id'] >= 0) & (chunk['dst_id'] >= 0)]

    # 레이블: 테스트 구간(DATE_OF_EVIL 이후) 공격자 이벤트만 = 1
    chunk['label'] = (
        chunk['user'].isin(ATTACKERS) &
        (chunk['ts'] >= DATE_OF_EVIL)
    ).astype(int)

    chunk['slice_ts'] = (chunk['ts'] // DELTA) * DELTA

    for slice_ts, group in chunk.groupby('slice_ts'):
        key = int(slice_ts)
        lines = (
            group['ts'].astype(str) + ',' +
            group['src_id'].astype(str) + ',' +
            group['dst_id'].astype(str) + ',' +
            group['label'].astype(str) + '\n'
        ).tolist()
        if key not in slice_data:
            slice_data[key] = []
        slice_data[key].extend(lines)
        total_events  += len(group)
        n_atk = int(group['label'].sum())
        attack_events += n_atk
        if n_atk > 0:
            attack_slices.add(key)

print('  슬라이스 파일 쓰기...')
for slice_ts in tqdm(sorted(slice_data.keys()), desc='파일 쓰기'):
    fpath = os.path.join(DST_DIR, f'{slice_ts}.txt')
    with open(fpath, 'w') as f:
        f.writelines(slice_data[slice_ts])

print(f'  총 이벤트:          {total_events:,}')
print(f'  공격 이벤트:         {attack_events:,}')
print(f'  슬라이스 수:         {len(slice_data):,}')
print(f'  공격 포함 슬라이스:  {len(attack_slices)}')

# ── Step 5: device.csv → flows 슬라이스 생성 ─────────────────────────────────
print('\n[Step 5] flows 슬라이스 생성 (device 기반)...')

flows_data  = {}
dev_events  = 0

dev_chunks = pd.read_csv(os.path.join(CERT_DIR, 'device.csv'),
                          usecols=['date', 'user', 'pc', 'activity'],
                          chunksize=chunksize)

# device flows: src=user_id, dst=첫번째 도메인 노드(dummy)
# pc는 nmap에 없으므로 dummy 노드 사용
# device 이벤트의 connect/disconnect 횟수를 flows feature로 활용

for chunk in tqdm(dev_chunks, desc='device→flows 처리'):
    chunk = chunk.copy()
    chunk['ts_dt']  = pd.to_datetime(chunk['date'], format='%m/%d/%Y %H:%M:%S')
    chunk['ts']     = chunk['ts_dt'].view(np.int64) // 10**9 - start_epoch
    chunk['src_id'] = chunk['user'].map(user_to_id).fillna(-1).astype(int)
    chunk['dst_id'] = len(users)  # dummy 노드 (첫 번째 도메인)
    chunk['is_connect']    = (chunk['activity'] == 'Connect').astype(int)
    chunk['is_disconnect'] = (chunk['activity'] == 'Disconnect').astype(int)

    chunk = chunk[(chunk['ts'] >= 0) & (chunk['src_id'] >= 0) & (chunk['dst_id'] >= 0)]
    chunk['slice_ts'] = (chunk['ts'] // DELTA) * DELTA

    for slice_ts, group in chunk.groupby('slice_ts'):
        key = int(slice_ts)
        lines = (
            group['ts'].astype(str) + ',' +
            group['src_id'].astype(str) + ',' +
            group['dst_id'].astype(str) + ',' +
            group['is_connect'].astype(str) + ',' +
            group['is_disconnect'].astype(str) + '\n'
        ).tolist()
        if key not in flows_data:
            flows_data[key] = []
        flows_data[key].extend(lines)
        dev_events += len(group)

print('  flows 파일 쓰기...')
for slice_ts in tqdm(sorted(flows_data.keys()), desc='flows 파일 쓰기'):
    fpath = os.path.join(DST_DIR, 'flows', f'{slice_ts}.txt')
    with open(fpath, 'w') as f:
        f.writelines(flows_data[slice_ts])

print(f'  device 이벤트:   {dev_events:,}')
print(f'  flows 슬라이스:  {len(flows_data):,}')

# ── Step 6: 메타 정보 저장 ────────────────────────────────────────────────────
print('\n[Step 6] 메타 정보 저장...')
meta = {
    'start_epoch':         start_epoch,
    'end_epoch':           end_epoch,
    'DATE_OF_EVIL':        DATE_OF_EVIL,
    'total_duration':      end_epoch - start_epoch,
    'num_users':           len(users),
    'num_domains':         len(domains),
    'num_nodes':           len(nmap),
    'total_events':        total_events,
    'attack_events':       attack_events,
    'attackers':           ATTACKERS,
    'attacker_windows':    attacker_windows,
    'DELTA':               DELTA,
    'FILE_DELTA':          FILE_DELTA,
    'modality':            'http+device',
}
pickle.dump(meta, open(os.path.join(DST_DIR, 'meta.pkl'), 'wb'))

print('\n===== 전처리 완료 =====')
print(f'  노드 수:            {len(nmap):,} (사용자 {len(users)} + 도메인 {len(domains)})')
print(f'  슬라이스 수:         {len(slice_data):,} (1일 단위)')
print(f'  공격 포함 슬라이스:  {len(attack_slices)}')
print(f'  총 이벤트:           {total_events:,}')
print(f'  공격 이벤트:         {attack_events:,}')
print(f'  DATE_OF_EVIL:        {DATE_OF_EVIL}초 ({DATE_OF_EVIL//86400}일)')
print(f'  출력 경로:            {DST_DIR}')
