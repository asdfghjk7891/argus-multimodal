# PicoDomain 전처리.
# 주 모달리티: Zeek kerberos/ntlm 로그, 보조 모달리티: conn.log (uid로 연결)
# 출력: DST_DIR/<ts>.txt (ts, src, dst, label), DST_DIR/flows/<ts>.txt (ts, src, dst, dur, ob, rb, op, rp),
#       nmap.pkl, meta.pkl

import os
import json
import pickle
import glob
import pandas as pd
from datetime import datetime, timezone
from collections import defaultdict
from tqdm import tqdm

PICO_DIR  = './PicoDomain/Zeek_Logs/'
RED_LOG   = './PicoDomain/Red Log.xlsx'
DST_DIR   = './data/picodomain/'
DELTA     = 3600   # 1시간

DAYS = ['2019-07-19', '2019-07-20', '2019-07-21']

os.makedirs(DST_DIR, exist_ok=True)
os.makedirs(os.path.join(DST_DIR, 'flows'), exist_ok=True)

# 1. IP -> 노드 ID
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

# 2. 공격 시작 시각
print('\n[Step 2] 공격 레이블 로드...')

ATTACK_HOSTS_IP = {'10.99.99.152', '10.99.99.30'}  # HR-WIN7-1, HR-WIN7-2

df_red = pd.read_excel(RED_LOG)

# Red Log 기준 공격 시작: 2019-07-19 18:08:00 UTC
ATTACK_START_UTC = datetime(2019, 7, 19, 18, 8, 0, tzinfo=timezone.utc)
ATTACK_START_TS  = int(ATTACK_START_UTC.timestamp())

DATA_START_UTC = datetime(2019, 7, 19, 0, 0, 0, tzinfo=timezone.utc)
DATA_START_TS  = int(DATA_START_UTC.timestamp())

DATA_END_UTC = datetime(2019, 7, 22, 0, 0, 0, tzinfo=timezone.utc)
DATA_END_TS  = int(DATA_END_UTC.timestamp())

DATE_OF_EVIL = ATTACK_START_TS - DATA_START_TS  # 65,280초

print(f'  데이터 시작: {DATA_START_UTC} (epoch={DATA_START_TS})')
print(f'  공격 시작:   {ATTACK_START_UTC} (epoch={ATTACK_START_TS})')
print(f'  DATE_OF_EVIL: {DATE_OF_EVIL}초 ({DATE_OF_EVIL/3600:.1f}시간)')
print(f'  훈련: 0 ~ {DATE_OF_EVIL//3600}시간')
print(f'  테스트: {DATE_OF_EVIL//3600}시간 ~ {(DATA_END_TS-DATA_START_TS)//3600}시간')

# 3. conn.log: uid -> 플로우 통계
print('\n[Step 3] conn.log에서 uid→flows 해시맵 구축...')

uid_to_flows = {}  # uid: [duration, orig_bytes, resp_bytes, orig_pkts, resp_pkts]
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

# 4. kerberos + ntlm -> 인증 슬라이스
print('\n[Step 4] auth 슬라이스 생성...')

slice_data    = defaultdict(list)
flows_data    = defaultdict(list)
total_auth    = 0
attack_events = 0
attack_slices = set()

def parse_ts(ts_str):
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

                # 공격 호스트가 관여하고 공격 시작 이후면 공격
                label = 1 if (
                    (src_ip in ATTACK_HOSTS_IP or dst_ip in ATTACK_HOSTS_IP)
                    and ts >= DATE_OF_EVIL
                ) else 0

                slice_ts = (ts // DELTA) * DELTA
                slice_data[slice_ts].append(
                    f'{ts},{src_id},{dst_id},{label}\n'
                )

                # 같은 uid의 conn.log 통계를 플로우로 기록
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

for day in DAYS:
    for fname in tqdm(sorted(glob.glob(f'{PICO_DIR}{day}/kerberos.*.log')),
                      desc=f'{day} kerberos'):
        process_auth_file(fname, 'kerberos')

for day in DAYS:
    for fname in tqdm(sorted(glob.glob(f'{PICO_DIR}{day}/ntlm.*.log')),
                      desc=f'{day} ntlm'):
        process_auth_file(fname, 'ntlm')

# 5. 파일 쓰기
print('\n[Step 5] 슬라이스 파일 쓰기...')
for slice_ts in tqdm(sorted(slice_data.keys()), desc='auth 파일'):
    with open(os.path.join(DST_DIR, f'{slice_ts}.txt'), 'w') as f:
        f.writelines(slice_data[slice_ts])

for slice_ts in tqdm(sorted(flows_data.keys()), desc='flows 파일'):
    with open(os.path.join(DST_DIR, 'flows', f'{slice_ts}.txt'), 'w') as f:
        f.writelines(flows_data[slice_ts])

# 6. 메타 정보
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
