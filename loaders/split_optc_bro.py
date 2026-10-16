# OpTC ecar-bro + bro conn 전처리.
# 출력: DST/<ts>.txt (ts, src, dst, label), DST/flows/<ts>.txt (bro conn 통계), DST/nmap_bro.pkl
# 메모리 때문에 날짜별로 나눠 처리함.
import gzip
import json
import os
import pickle
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm

ECAR_BRO_DIR = './data/ecar-bro'
BRO_DIR      = './data/bro'
REDTEAM_FILE = './data/redteam_optc.txt'
OLD_NMAP     = './data/nmap.pkl'        # ARGUS OpTC 노드 목록
DST          = './data/optc_bro/'
DELTA        = 10000  # 파일 단위 (초)

os.makedirs(DST, exist_ok=True)
os.makedirs(os.path.join(DST, 'flows'), exist_ok=True)


def get_or_add(n, m, id_):
    if n not in m:
        m[n] = id_[0]
        id_[0] += 1
    return m[n]


def get_ecar_bro_files():
    file_list = []
    for split in ['benign', 'evaluation']:
        split_path = os.path.join(ECAR_BRO_DIR, split)
        if not os.path.isdir(split_path):
            continue
        for period in sorted(os.listdir(split_path)):
            period_path = os.path.join(split_path, period)
            if not os.path.isdir(period_path):
                continue
            for folder in sorted(os.listdir(period_path)):
                ecar_file = os.path.join(period_path, folder, 'ecarbro.json.gz')
                if os.path.exists(ecar_file):
                    file_list.append(ecar_file)
    return file_list


def epoch_to_date(epoch):
    return datetime.utcfromtimestamp(epoch).strftime('%Y-%m-%d')


def parse_ts(ts_str):
    try:
        dt = datetime.fromisoformat(ts_str)
        return int(dt.timestamp())
    except Exception:
        return None


# 1. 레드팀 hostname 쌍
print('[Step 1] 레드팀 이벤트 로드...')
with open(OLD_NMAP, 'rb') as f:
    old_nmap = pickle.load(f)

red_hostname_pairs = set()
with open(REDTEAM_FILE, 'r') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split(',')
        if len(parts) < 3:
            continue
        try:
            src_id = int(parts[1])
            dst_id = int(parts[2])
            src_num = int(old_nmap[src_id])
            dst_num = int(old_nmap[dst_id])
            src_h = f'SysClient{src_num:04d}.systemia.com'
            dst_h = f'SysClient{dst_num:04d}.systemia.com'
            red_hostname_pairs.add((src_h, dst_h))
        except Exception:
            continue

print(f'  레드팀 hostname 쌍: {len(red_hostname_pairs)}')

red_all_hosts = set()
for src_h, dst_h in red_hostname_pairs:
    red_all_hosts.add(src_h)
    red_all_hosts.add(dst_h)

print(f'  공격 관련 호스트 수: {len(red_all_hosts)}')

# ecar-bro와 auth_optc(ARGUS) 시간 오프셋 보정값. 공격 구간은 auth_optc 기준
AUTH_TO_ECAR_OFFSET = 8028
RED_TS_START = 573290
RED_TS_END   = 745983

# 2. IP -> hostname 매핑
print('[Step 2] IP→hostname 매핑 구축...')
ip_to_hostname = {}
ecar_files = get_ecar_bro_files()

for fpath in tqdm(ecar_files, desc='IP 매핑'):
    try:
        with gzip.open(fpath, 'rt', errors='replace') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    hostname = data.get('hostname', '')
                    props = data.get('properties', {})
                    src_ip = props.get('src_ip', '')
                    if hostname and src_ip and 'SysClient' in hostname:
                        ip_to_hostname[src_ip] = hostname
                except Exception:
                    continue
    except Exception:
        continue

print(f'  IP→hostname 매핑 수: {len(ip_to_hostname)}')


# 3. 날짜별 ecar-bro 파일 목록
print('[Step 3] ecar-bro 날짜 목록 파악...')
date_to_files = defaultdict(list)  # {'2019-09-17': [fpath1, fpath2, ...]}

for fpath in ecar_files:
    try:
        with gzip.open(fpath, 'rt', errors='replace') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    bro_uid = data.get('properties', {}).get('bro_uid', '')
                    if not bro_uid:
                        continue
                    hostname = data.get('hostname', '')
                    if 'SysClient' not in hostname:
                        continue
                    ts_str = data.get('timestamp', '')
                    epoch = parse_ts(ts_str)
                    if epoch is not None:
                        date_str = epoch_to_date(epoch)
                        date_to_files[date_str].append(fpath)
                        break
                except Exception:
                    continue
    except Exception:
        continue

dates_sorted = sorted(date_to_files.keys())
print(f'  날짜 수: {len(dates_sorted)}')
for d in dates_sorted:
    print(f'  {d}: {len(date_to_files[d])}개 파일')


# 4. 전체 시작 epoch
print('[Step 4] 전체 시작 epoch 파악...')
global_min_epoch = float('inf')

for fpath in ecar_files:
    try:
        with gzip.open(fpath, 'rt', errors='replace') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    bro_uid = data.get('properties', {}).get('bro_uid', '')
                    if not bro_uid:
                        continue
                    hostname = data.get('hostname', '')
                    if 'SysClient' not in hostname:
                        continue
                    ts_str = data.get('timestamp', '')
                    epoch = parse_ts(ts_str)
                    if epoch is not None and epoch < global_min_epoch:
                        global_min_epoch = epoch
                    break
                except Exception:
                    continue
    except Exception:
        continue

if global_min_epoch == float('inf'):
    raise ValueError('유효한 ecar-bro 이벤트를 찾지 못했습니다.')

start_epoch = global_min_epoch
print(f'  시작 epoch: {start_epoch} ({datetime.fromtimestamp(start_epoch)})')


# 5. 노드 매핑, 출력 파일
nmap_new = {}
nid = [0]

cur_time = 0
f_auth  = open(os.path.join(DST, f'{cur_time}.txt'), 'w')
f_flows = open(os.path.join(DST, 'flows', f'{cur_time}.txt'), 'w')


# 6. 날짜별 처리
print('[Step 6] 날짜별 처리 시작...')

for date_str in tqdm(dates_sorted, desc='날짜 처리'):
    files_today = date_to_files[date_str]

    # 그날 이벤트의 bro uid
    today_uids = set()
    for fpath in files_today:
        try:
            with gzip.open(fpath, 'rt', errors='replace') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        bro_uid = data.get('properties', {}).get('bro_uid', '')
                        if bro_uid:
                            today_uids.add(bro_uid)
                    except Exception:
                        continue
        except Exception:
            continue

    # 그날 bro conn에서 해당 uid의 통계만 읽음
    uid_to_stats = {}
    bro_date_path = os.path.join(BRO_DIR, date_str)

    if os.path.isdir(bro_date_path):
        conn_files = sorted([f for f in os.listdir(bro_date_path)
                              if f.startswith('conn.')])
        for fname in conn_files:
            fpath_conn = os.path.join(bro_date_path, fname)
            try:
                with gzip.open(fpath_conn, 'rt', errors='replace') as f:
                    for line in f:
                        if line.startswith('#'):
                            continue
                        parts = line.strip().split('\t')
                        if len(parts) < 19:
                            continue
                        uid = parts[1]
                        if uid not in today_uids:
                            continue
                        try:
                            duration = float(parts[8])  if parts[8]  != '-' else 0.0
                            orig_b   = int(parts[9])    if parts[9]  != '-' else 0
                            resp_b   = int(parts[10])   if parts[10] != '-' else 0
                            orig_p   = int(parts[16])   if parts[16] != '-' else 0
                            resp_p   = int(parts[18])   if parts[18] != '-' else 0
                            dest_p   = int(parts[5])    if parts[5]  != '-' else 0
                            uid_to_stats[uid] = (duration, orig_b, resp_b,
                                                  orig_p,   resp_p,  dest_p)
                        except Exception:
                            continue
            except Exception:
                continue

    del today_uids

    # ecar-bro 이벤트를 다시 읽어 슬라이스 생성
    today_events = []
    for fpath in files_today:
        try:
            with gzip.open(fpath, 'rt', errors='replace') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        props = data.get('properties', {})
                        bro_uid = props.get('bro_uid', '')
                        if not bro_uid:
                            continue
                        src_hostname = data.get('hostname', '')
                        if 'SysClient' not in src_hostname:
                            continue
                        dest_ip = props.get('dest_ip', '')
                        dst_hostname = ip_to_hostname.get(dest_ip, dest_ip)
                        ts_str = data.get('timestamp', '')
                        epoch = parse_ts(ts_str)
                        if epoch is None:
                            continue
                        # 공격 호스트가 관여하고 공격 구간 안이면 공격
                        rel_ts = epoch - start_epoch - AUTH_TO_ECAR_OFFSET
                        label = 1 if (
                            (src_hostname in red_all_hosts or
                             dst_hostname in red_all_hosts)
                            and RED_TS_START <= rel_ts <= RED_TS_END
                        ) else 0
                        today_events.append(
                            (epoch, src_hostname, dst_hostname, label, bro_uid)
                        )
                    except Exception:
                        continue
        except Exception:
            continue

    today_events.sort(key=lambda x: x[0])

    for epoch, src_h, dst_h, label, bro_uid in today_events:
        ts = epoch - start_epoch
        if ts < 0:
            ts = 0

        src_id = get_or_add(src_h, nmap_new, nid)
        dst_id = get_or_add(dst_h, nmap_new, nid)

        correct_slice = (ts // DELTA) * DELTA
        if correct_slice != cur_time:
            f_auth.close()
            f_flows.close()
            cur_time = correct_slice
            f_auth  = open(os.path.join(DST, f'{cur_time}.txt'), 'w')
            f_flows = open(os.path.join(DST, 'flows', f'{cur_time}.txt'), 'w')

        f_auth.write(f'{ts},{src_id},{dst_id},{label}\n')

        if bro_uid in uid_to_stats:
            dur, ob, rb, op, rp, dp = uid_to_stats[bro_uid]
            f_flows.write(f'{ts},{src_id},{dst_id},'
                          f'{dur},{ob},{rb},{op},{rp},{dp}\n')

    del uid_to_stats
    del today_events

f_auth.close()
f_flows.close()


# 7. nmap 저장
print('[Step 7] nmap 저장...')
nmap_rev = [None] * (max(nmap_new.values()) + 1)
for k, v in nmap_new.items():
    nmap_rev[v] = k

with open(os.path.join(DST, 'nmap_bro.pkl'), 'wb') as f:
    pickle.dump(nmap_rev, f, protocol=pickle.HIGHEST_PROTOCOL)

print('\n===== 전처리 완료 =====')
print(f'  노드 수: {len(nmap_new)}')
print(f'  슬라이스 수: {cur_time // DELTA + 1}')
print(f'  출력 경로: {DST}')

attack_slices = 0
total_attack_events = 0
for fname in sorted(os.listdir(DST)):
    if not fname.endswith('.txt'):
        continue
    has_attack = False
    with open(os.path.join(DST, fname), 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            # 형식: ts, src, dst, label
            if len(parts) >= 4 and parts[3] == '1':
                total_attack_events += 1
                has_attack = True
    if has_attack:
        attack_slices += 1

print(f'  공격 포함 슬라이스 수: {attack_slices}')
print(f'  총 공격 이벤트 수: {total_attack_events:,}')
