"""
split_picodomain.py
PicoDomain 데이터셋을 ARGUS LANL 형식으로 전처리

입력:
  Zeek_Logs/2019-07-19~21/kerberos.*.log  ← auth (주 모달리티)
  Zeek_Logs/2019-07-19~21/ntlm.*.log      ← auth (kerberos 보완)
  Zeek_Logs/2019-07-19~21/conn.*.log      ← flows (uid 매핑)
  Red Log.xlsx                             ← 공격 레이블

출력:
  data/picodomain/[ts].txt        : ts, src_id, dst_id, label
  data/picodomain/flows/[ts].txt  : ts, src_id, dst_id, dur, ob, rb, op, rp
  data/picodomain/nmap.pkl        : 노드ID 매핑
  data/picodomain/meta.pkl        : 메타 정보
"""

import os
import json
import pickle
import glob
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from collections import defaultdict
from tqdm import tqdm

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
PICO_DIR  = 'C:/Users/user/Desktop/PicoDomain/Zeek_Logs/'
RED_LOG   = 'C:/Users/user/Desktop/PicoDomain/Red Log.xlsx'
DST_DIR   = 'C:/Users/user/Desktop/Argus/data/picodomain/'
DELTA     = 3600   # 1시간 단위 슬라이스

DAYS = ['2019-07-19', '2019-07-20', '2019-07-21']

os.makedirs(DST_DIR, exist_ok=True)
os.makedirs(os.path.join(DST_DIR, 'flows'), exist_ok=True)

# ── Step 1: IP → 노드ID 매핑 ─────────────────────────────────────────────────
print('[Step 1] 노드 ID 매핑 구축...')

IP_TO_HOST = {
    '10.99.99.5'  : 'CORP-DC',
    '10.99.99.27' : 'RND-WIN10-2',
    '10.99.99.29' : 'RND-WIN10-1',
    '10.99.99.30' : 'HR-WIN7-2',
    '10.99.99.152': 'HR-WIN7-1',
    '10.99.99.160': 'SUPERSECRETXP',
}
hosts  = sorted(IP_TO_HOST.values())
ip_ids = {ip: hosts.index(IP_TO_HOST[ip]) for ip in IP_TO_HOST}
nmap   = hosts

pickle.dump(nmap, open(os.path.join(DST_DIR, 'nmap.pkl'), 'wb'))
print(f'  노드 수: {len(nmap)}')
for ip, host in sorted(IP_TO_HOST.items()):
    print(f'  {ip} → {hosts.index(host)}: {host}')

# ── Step 2: 공격 시간 범위 파싱 ───────────────────────────────────────────────
print('\n[Step 2] 공격 레이블 로드...')

ATTACK_HOSTS_IP = {'10.99.99.152', '10.99.99.30'}  # HR-WIN7-1, HR-WIN7-2

df_red = pd.read_excel(RED_LOG)

# 공격 시작: 2019-07-19 18:08:00 UTC
ATTACK_START_UTC = datetime(2019, 7, 19, 18, 8, 0, tzinfo=timezone.utc)
ATTACK_START_TS  = int(ATTACK_START_UTC.timestamp())

# 데이터 시작: 2019-07-19 00:00:00 UTC
DATA_START_UTC = datetime(2019, 7, 19, 0, 0, 0, tzinfo=timezone.utc)
DATA_START_TS  = int(DATA_START_UTC.timestamp())

DATA_END_UTC = datetime(2019, 7, 22, 0, 0, 0, tzinfo=timezone.utc)
DATA_END_TS  = int(DATA_END_UTC.timestamp())

DATE_OF_EVIL = ATTACK_START_TS - DATA_START_TS  # 65,280초 (18.13시간)

print(f'  데이터 시작: {DATA_START_UTC} (epoch={DATA_START_TS})')
print(f'  공격 시작:   {ATTACK_START_UTC} (epoch={ATTACK_START_TS})')
print(f'  DATE_OF_EVIL: {DATE_OF_EVIL}초 ({DATE_OF_EVIL/3600:.1f}시간)')
print(f'  훈련: 0 ~ {DATE_OF_EVIL//3600}시간')
print(f'  테스트: {DATE_OF_EVIL//3600}시간 ~ {(DATA_END_TS-DATA_START_TS)//3600}시간')

# ── Step 3: conn.log → uid→flows 해시맵 구축 ─────────────────────────────────
print('\n[Step 3] conn.log에서 uid→flows 해시맵 구축...')

uid_to_flows = {}  # {uid: [duration, orig_bytes, resp_bytes, orig_pkts, resp_pkts]}
total_conn = 0

for day in DAYS:
    for fname in sorted(glob.glob(f'{PICO_DIR}{day}/conn.*.log')):
        with open(fname, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    uid      = d.get('uid', '')
                    dur      = float(d.get('duration', 0) or 0)
                    ob       = int(d.get('orig_bytes', 0) or 0)
                    rb       = int(d.get('resp_bytes', 0) or 0)
                    op       = int(d.get('orig_pkts', 0) or 0)
                    rp       = int(d.get('resp_pkts', 0) or 0)
                    if uid:
                        uid_to_flows[uid] = [dur, ob, rb, op, rp]
                    total_conn += 1
                except:
                    continue

print(f'  총 conn 이벤트: {total_conn:,}')
print(f'  uid→flows 해시맵: {len(uid_to_flows):,}개')

# ── Step 4: kerberos + ntlm → auth 슬라이스 생성 ─────────────────────────────
print('\n[Step 4] auth 슬라이스 생성...')

slice_data    = defaultdict(list)
flows_data    = defaultdict(list)
total_auth    = 0
attack_events = 0
attack_slices = set()

def parse_ts(ts_str):
    """ISO 8601 타임스탬프 → epoch 상대시간"""
    try:
        dt = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
        return int(dt.timestamp()) - DATA_START_TS
    except:
        return None

def process_auth_file(fname, proto='kerberos'):
    global total_auth, attack_events
    with open(fname, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                ts     = parse_ts(d.get('ts', ''))
                src_ip = d.get('id.orig_h', '')
                dst_ip = d.get('id.resp_h', '')
                uid    = d.get('uid', '')

                if ts is None or ts < 0:
                    continue
                if src_ip not in ip_ids or dst_ip not in ip_ids:
                    continue
                if src_ip == dst_ip:
                    continue

                src_id = ip_ids[src_ip]
                dst_id = ip_ids[dst_ip]

                # 레이블: 공격 호스트 관여 + 공격 시작 이후
                label = 1 if (
                    (src_ip in ATTACK_HOSTS_IP or dst_ip in ATTACK_HOSTS_IP)
                    and ts >= DATE_OF_EVIL
                ) else 0

                slice_ts = (ts // DELTA) * DELTA
                slice_data[slice_ts].append(
                    f'{ts},{src_id},{dst_id},{label}\n'
                )

                # flows 매핑 (uid로 conn.log와 연결)
                if uid in uid_to_flows:
                    dur, ob, rb, op, rp = uid_to_flows[uid]
                    flows_data[slice_ts].append(
                        f'{ts},{src_id},{dst_id},{dur},{ob},{rb},{op},{rp}\n'
                    )

                total_auth += 1
                if label == 1:
                    attack_events += 1
                    attack_slices.add(slice_ts)

            except Exception as e:
                continue

# kerberos 처리
for day in DAYS:
    for fname in tqdm(sorted(glob.glob(f'{PICO_DIR}{day}/kerberos.*.log')),
                      desc=f'{day} kerberos'):
        process_auth_file(fname, 'kerberos')

# ntlm 처리
for day in DAYS:
    for fname in tqdm(sorted(glob.glob(f'{PICO_DIR}{day}/ntlm.*.log')),
                      desc=f'{day} ntlm'):
        process_auth_file(fname, 'ntlm')

# ── Step 5: 파일 쓰기 ─────────────────────────────────────────────────────────
print('\n[Step 5] 슬라이스 파일 쓰기...')
for slice_ts in tqdm(sorted(slice_data.keys()), desc='auth 파일'):
    with open(os.path.join(DST_DIR, f'{slice_ts}.txt'), 'w') as f:
        f.writelines(slice_data[slice_ts])

for slice_ts in tqdm(sorted(flows_data.keys()), desc='flows 파일'):
    with open(os.path.join(DST_DIR, 'flows', f'{slice_ts}.txt'), 'w') as f:
        f.writelines(flows_data[slice_ts])

# ── Step 6: 메타 정보 저장 ────────────────────────────────────────────────────
print('\n[Step 6] 메타 정보 저장...')
meta = {
    'start_epoch'   : DATA_START_TS,
    'end_epoch'     : DATA_END_TS,
    'DATE_OF_EVIL'  : DATE_OF_EVIL,
    'total_duration': DATA_END_TS - DATA_START_TS,
    'num_nodes'     : len(nmap),
    'total_auth'    : total_auth,
    'attack_events' : attack_events,
    'attack_hosts'  : list(ATTACK_HOSTS_IP),
    'DELTA'         : DELTA,
    'FILE_DELTA'    : DELTA,
    'ip_to_host'    : IP_TO_HOST,
}
pickle.dump(meta, open(os.path.join(DST_DIR, 'meta.pkl'), 'wb'))

print('\n===== 전처리 완료 =====')
print(f'  노드 수:            {len(nmap)}')
print(f'  총 auth 이벤트:     {total_auth:,}')
print(f'  공격 이벤트:         {attack_events:,}')
print(f'  슬라이스 수:         {len(slice_data)}')
print(f'  공격 포함 슬라이스:  {len(attack_slices)}')
print(f'  flows 슬라이스:      {len(flows_data)}')
print(f'  DATE_OF_EVIL:        {DATE_OF_EVIL}초 ({DATE_OF_EVIL/3600:.1f}시간)')
print(f'  출력 경로:            {DST_DIR}')
