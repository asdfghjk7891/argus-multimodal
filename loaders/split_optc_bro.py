"""
split_optc_bro.py (메모리 최적화 버전 - 날짜별 완전 분리)
ecar-bro + bro conn 데이터를 전처리하여 멀티모달 슬라이스 파일 생성

메모리 전략:
  - ecar-bro를 날짜별로 2번 읽음 (uid 수집 → 슬라이스 생성)
  - bro conn도 날짜별로 읽고 처리 후 즉시 해제
  - 한 번에 메모리에 올라가는 데이터: 하루치만
"""
import gzip
import json
import os
import pickle
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm

# ── 경로 설정 ──────────────────────────────────────────────
ECAR_BRO_DIR = 'C:/Users/user/Desktop/Argus/data/ecar-bro'
BRO_DIR      = 'C:/Users/user/Desktop/Argus/data/bro'
REDTEAM_FILE = 'C:/Users/user/Desktop/Argus/data/redteam_optc.txt'
OLD_NMAP     = 'C:/Users/user/Desktop/Argus/data/nmap.pkl'
DST          = 'C:/Users/user/Desktop/Argus/data/optc_bro/'
DELTA        = 10000  # 슬라이스 크기 (초)
# ──────────────────────────────────────────────────────────

os.makedirs(DST, exist_ok=True)
os.makedirs(os.path.join(DST, 'flows'), exist_ok=True)


# ── 유틸 함수 ─────────────────────────────────────────────
def get_or_add(n, m, id_):
    if n not in m:
        m[n] = id_[0]
        id_[0] += 1
    return m[n]


def get_ecar_bro_files():
    """ecar-bro 폴더의 모든 ecarbro.json.gz 파일 경로 반환"""
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
    """epoch → 'YYYY-MM-DD' (UTC 기준)"""
    return datetime.utcfromtimestamp(epoch).strftime('%Y-%m-%d')


def parse_ts(ts_str):
    """ISO 타임스탬프 → epoch (정수)"""
    try:
        dt = datetime.fromisoformat(ts_str)
        return int(dt.timestamp())
    except Exception:
        return None


# ── Step 1: 레드팀 hostname 쌍 로드 ───────────────────────
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

# 공격 관련 모든 호스트 집합
red_all_hosts = set()
for src_h, dst_h in red_hostname_pairs:
    red_all_hosts.add(src_h)
    red_all_hosts.add(dst_h)

print(f'  공격 관련 호스트 수: {len(red_all_hosts)}')
print(f'  샘플: {sorted(red_all_hosts)[:5]}')

# 공격 시간 범위 (auth_optc 기준 ts → ecar-bro epoch 변환)
# 오프셋: ecar-bro 시작 - auth_optc 시작 = 8028초
# auth_optc redteam: 573290 ~ 745983
# 단, start_epoch는 Step 4에서 결정되므로 여기서는 상수로 설정
AUTH_TO_ECAR_OFFSET = 8028
RED_TS_START = 573290  # auth_optc 기준
RED_TS_END   = 745983  # auth_optc 기준

# ── Step 2: IP→hostname 매핑 구축 ─────────────────────────
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


# ── Step 3: ecar-bro 날짜 목록 파악 ───────────────────────
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


# ── Step 4: 전체 시작 epoch 파악 ──────────────────────────
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
                    break  # 파일당 첫 유효 이벤트만
                except Exception:
                    continue
    except Exception:
        continue

# 시작 epoch 못 찾은 경우 예외처리
if global_min_epoch == float('inf'):
    raise ValueError('유효한 ecar-bro 이벤트를 찾지 못했습니다.')

start_epoch = global_min_epoch
print(f'  시작 epoch: {start_epoch} ({datetime.fromtimestamp(start_epoch)})')


# ── Step 5: 노드 매핑 초기화 ──────────────────────────────
nmap_new = {}
nid = [0]

# 슬라이스 파일 초기화
cur_time = 0
f_auth  = open(os.path.join(DST, f'{cur_time}.txt'), 'w')
f_flows = open(os.path.join(DST, 'flows', f'{cur_time}.txt'), 'w')


# ── Step 6: 날짜별 처리 ───────────────────────────────────
print('[Step 6] 날짜별 처리 시작...')

for date_str in tqdm(dates_sorted, desc='날짜 처리'):
    files_today = date_to_files[date_str]

    # 6-1. 오늘 날짜의 uid 목록 수집
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

    # 6-2. 해당 날짜 bro conn 로드 (today_uids만 필터링)
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

    # today_uids 메모리 해제
    del today_uids

    # 6-3. 오늘 날짜 ecar-bro 재파싱 → 슬라이스 생성
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
                        # 공격 호스트 관여 여부 + 공격 시간 범위 체크
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

    # 시간순 정렬
    today_events.sort(key=lambda x: x[0])

    for epoch, src_h, dst_h, label, bro_uid in today_events:
        ts = epoch - start_epoch
        if ts < 0:
            ts = 0

        src_id = get_or_add(src_h, nmap_new, nid)
        dst_id = get_or_add(dst_h, nmap_new, nid)

        # 슬라이스 분할: ts 기준으로 올바른 파일 직접 선택
        correct_slice = (ts // DELTA) * DELTA
        if correct_slice != cur_time:
            f_auth.close()
            f_flows.close()
            cur_time = correct_slice
            f_auth  = open(os.path.join(DST, f'{cur_time}.txt'), 'w')
            f_flows = open(os.path.join(DST, 'flows', f'{cur_time}.txt'), 'w')

        # auth 슬라이스 저장
        f_auth.write(f'{ts},{src_id},{dst_id},{label}\n')

        # flows 슬라이스 저장
        if bro_uid in uid_to_stats:
            dur, ob, rb, op, rp, dp = uid_to_stats[bro_uid]
            f_flows.write(f'{ts},{src_id},{dst_id},'
                          f'{dur},{ob},{rb},{op},{rp},{dp}\n')

    # 메모리 해제
    del uid_to_stats
    del today_events

f_auth.close()
f_flows.close()


# ── Step 7: nmap 저장 ─────────────────────────────────────
print('[Step 7] nmap 저장...')
nmap_rev = [None] * (max(nmap_new.values()) + 1)
for k, v in nmap_new.items():
    nmap_rev[v] = k

with open(os.path.join(DST, 'nmap_bro.pkl'), 'wb') as f:
    pickle.dump(nmap_rev, f, protocol=pickle.HIGHEST_PROTOCOL)

# ── 완료 보고 ─────────────────────────────────────────────
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
