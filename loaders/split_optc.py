"""
split_optc_bro.py
ecar-bro + bro conn 데이터를 전처리하여 멀티모달 슬라이스 파일 생성

출력:
  data/optc_bro/         ← auth 슬라이스 (ts, src, dst, label)
  data/optc_bro/flows/   ← flows 슬라이스 (ts, src, dst, duration, orig_b, resp_b, orig_p, resp_p, dest_port)
  data/optc_bro/nmap_bro.pkl ← hostname → 노드ID 매핑
"""
import gzip, json, os, pickle
from datetime import datetime
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
os.makedirs(DST + 'flows/', exist_ok=True)

# ── Step 1: 레드팀 hostname 쌍 로드 ───────────────────────
print('[Step 1] 레드팀 이벤트 로드...')
with open(OLD_NMAP, 'rb') as f:
    old_nmap = pickle.load(f)

red_hostname_pairs = set()
with open(REDTEAM_FILE, 'r') as f:
    for line in f:
        parts = line.strip().split(',')
        if len(parts) < 3:
            continue
        src_id  = int(parts[1])
        dst_id  = int(parts[2])
        src_num = old_nmap[src_id]
        dst_num = old_nmap[dst_id]
        src_h   = f'SysClient{int(src_num):04d}.systemia.com'
        dst_h   = f'SysClient{int(dst_num):04d}.systemia.com'
        red_hostname_pairs.add((src_h, dst_h))

print(f'  레드팀 hostname 쌍: {len(red_hostname_pairs)}')

# ── Step 2: IP → hostname 매핑 구축 ───────────────────────
print('[Step 2] IP→hostname 매핑 구축...')
ip_to_hostname = {}

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
            if not os.path.exists(ecar_file):
                continue
            with gzip.open(ecar_file, 'rt', errors='replace') as f:
                for line in f:
                    try:
                        data     = json.loads(line)
                        hostname = data.get('hostname', '')
                        props    = data.get('properties', {})
                        src_ip   = props.get('src_ip', '')
                        if hostname and src_ip and 'SysClient' in hostname:
                            ip_to_hostname[src_ip] = hostname
                    except:
                        continue

print(f'  IP→hostname 매핑 수: {len(ip_to_hostname)}')

# ── Step 3: bro conn 전체 로드 → uid_to_stats ─────────────
print('[Step 3] bro conn 파일 로드...')
uid_to_stats = {}
dates = sorted([d for d in os.listdir(BRO_DIR)
                if os.path.isdir(os.path.join(BRO_DIR, d))])

for date in tqdm(dates, desc='날짜 처리'):
    date_path = os.path.join(BRO_DIR, date)
    for fname in sorted(os.listdir(date_path)):
        if not fname.startswith('conn.'):
            continue
        fpath = os.path.join(date_path, fname)
        try:
            with gzip.open(fpath, 'rt', errors='replace') as f:
                for line in f:
                    if line.startswith('#'):
                        continue
                    parts = line.strip().split('\t')
                    if len(parts) < 19:
                        continue
                    uid      = parts[1]
                    duration = float(parts[8])  if parts[8]  != '-' else 0.0
                    orig_b   = int(parts[9])    if parts[9]  != '-' else 0
                    resp_b   = int(parts[10])   if parts[10] != '-' else 0
                    orig_p   = int(parts[16])   if parts[16] != '-' else 0
                    resp_p   = int(parts[18])   if parts[18] != '-' else 0
                    dest_p   = int(parts[5])    if parts[5]  != '-' else 0
                    uid_to_stats[uid] = (duration, orig_b, resp_b,
                                         orig_p,   resp_p,  dest_p)
        except:
            continue

print(f'  uid 수: {len(uid_to_stats):,}')

# ── Step 4: ecar-bro 파싱 → 이벤트 수집 ──────────────────
print('[Step 4] ecar-bro 파싱 및 이벤트 수집...')
events = []  # (epoch, src_h, dst_h, label, bro_uid)

for split in ['benign', 'evaluation']:
    split_path = os.path.join(ECAR_BRO_DIR, split)
    if not os.path.isdir(split_path):
        continue
    for period in tqdm(sorted(os.listdir(split_path)), desc=f'{split}'):
        period_path = os.path.join(split_path, period)
        if not os.path.isdir(period_path):
            continue
        for folder in sorted(os.listdir(period_path)):
            ecar_file = os.path.join(period_path, folder, 'ecarbro.json.gz')
            if not os.path.exists(ecar_file):
                continue
            with gzip.open(ecar_file, 'rt', errors='replace') as f:
                for line in f:
                    try:
                        data     = json.loads(line)
                        props    = data.get('properties', {})
                        bro_uid  = props.get('bro_uid', '')
                        if not bro_uid:
                            continue
                        src_hostname = data.get('hostname', '')
                        if 'SysClient' not in src_hostname:
                            continue
                        dest_ip  = props.get('dest_ip', '')
                        dst_hostname = ip_to_hostname.get(dest_ip, dest_ip)

                        ts_str = data.get('timestamp', '')
                        dt     = datetime.fromisoformat(ts_str)
                        epoch  = int(dt.timestamp())

                        label = 1 if (src_hostname, dst_hostname) \
                                     in red_hostname_pairs else 0

                        events.append((epoch, src_hostname,
                                       dst_hostname, label, bro_uid))
                    except:
                        continue

print(f'  전체 이벤트 수: {len(events):,}')
attack_count = sum(1 for e in events if e[3] == 1)
print(f'  공격 이벤트 수: {attack_count:,}')

# ── Step 5: 타임스탬프 정규화 (시작=0) ────────────────────
print('[Step 5] 타임스탬프 정규화...')
events.sort(key=lambda x: x[0])
start_epoch = events[0][0]
events = [(e[0] - start_epoch, e[1], e[2], e[3], e[4])
          for e in events]
print(f'  시작 epoch: {start_epoch} ({datetime.fromtimestamp(start_epoch)})')
print(f'  정규화 후 ts 범위: 0 ~ {events[-1][0]}')

# ── Step 6: 노드 매핑 생성 ────────────────────────────────
print('[Step 6] 노드 매핑 생성...')
nmap_new = {}
nid = [0]

def get_or_add(n, m, id_):
    if n not in m:
        m[n] = id_[0]
        id_[0] += 1
    return m[n]

# ── Step 7: 슬라이스 파일 생성 ────────────────────────────
print('[Step 7] 슬라이스 파일 생성...')
cur_time  = 0
f_auth    = open(DST + f'{cur_time}.txt', 'w')
f_flows   = open(DST + f'flows/{cur_time}.txt', 'w')

for ts, src_h, dst_h, label, bro_uid in tqdm(events, desc='슬라이스 생성'):
    src_id = get_or_add(src_h, nmap_new, nid)
    dst_id = get_or_add(dst_h, nmap_new, nid)

    # auth 슬라이스
    f_auth.write(f'{ts},{src_id},{dst_id},{label}\n')

    # flows 슬라이스 (bro conn 통계)
    if bro_uid in uid_to_stats:
        dur, ob, rb, op, rp, dp = uid_to_stats[bro_uid]
        f_flows.write(f'{ts},{src_id},{dst_id},{dur},{ob},{rb},{op},{rp},{dp}\n')

    # 슬라이스 분할
    if ts >= cur_time + DELTA:
        f_auth.close()
        f_flows.close()
        cur_time += DELTA
        f_auth  = open(DST + f'{cur_time}.txt', 'w')
        f_flows = open(DST + f'flows/{cur_time}.txt', 'w')

f_auth.close()
f_flows.close()

# ── Step 8: nmap 저장 ─────────────────────────────────────
print('[Step 8] nmap 저장...')
nmap_rev = [None] * (max(nmap_new.values()) + 1)
for k, v in nmap_new.items():
    nmap_rev[v] = k

with open(DST + 'nmap_bro.pkl', 'wb') as f:
    pickle.dump(nmap_rev, f, protocol=pickle.HIGHEST_PROTOCOL)

print(f'\n완료!')
print(f'  노드 수: {len(nmap_new)}')
print(f'  슬라이스 수: {cur_time // DELTA + 1}')
print(f'  출력 경로: {DST}')