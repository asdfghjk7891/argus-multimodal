from copy import deepcopy
import os
import pickle

import torch
from tqdm import tqdm
import numpy as np

from .tdata import TData
from .load_utils import edge_tv_split, std_edge_w, standardized, std_edge_a

# ── 상수 설정 ─────────────────────────────────────────────────────────────────
PICO_FOLDER = 'C:/Users/user/Desktop/Argus/data/picodomain/'
assert PICO_FOLDER, 'Please fill in PICO_FOLDER in loaders/load_picodomain.py'

FILE_DELTA = 3600   # 1시간 단위

# meta.pkl에서 로드 (전처리 완료 후 자동 갱신)
DATE_OF_EVIL_PICO = 65280   # 18.1시간 (2019-07-19 18:08 UTC)

TIMES = {
    'test': 259200,   # 72시간 (3일 전체)
    'all' : 259200,
}

def load_meta():
    meta_path = PICO_FOLDER + 'meta.pkl'
    if os.path.exists(meta_path):
        return pickle.load(open(meta_path, 'rb'))
    return None

# ── 유틸 함수 ─────────────────────────────────────────────────────────────────
def empty_pico(use_flows=False):
    return make_data_obj(None, [], None, None, None, use_flows=use_flows)

def load_pico_dist(start=0, end=None, delta=3600, is_test=False,
                   use_flows=False, ew_fn=std_edge_w, ea_fn=std_edge_a):
    if start is None or end is None:
        return empty_pico(use_flows)
    return load_partial_pico(start, end, delta, is_test, use_flows, ew_fn, ea_fn)

def make_data_obj(cur_slice, eis, ys, ew_fn, ea_fn,
                  ews=None, eas=None, use_flows=False, **kwargs):
    if 'node_map' in kwargs:
        nm = kwargs['node_map']
    else:
        nm = pickle.load(open(PICO_FOLDER + 'nmap.pkl', 'rb'))

    cl_cnt = len(nm)
    x = torch.eye(cl_cnt + 1)

    eis_t = []
    masks = []
    for i in range(len(eis)):
        ei = torch.tensor(eis[i])
        eis_t.append(ei)
        if isinstance(ys, None.__class__):
            # PicoDomain은 슬라이스당 엣지가 적어서 v_size 크게 설정
            masks.append(edge_tv_split(ei, v_size=0.3)[0])

    if not isinstance(ews, None.__class__):
        cnt = deepcopy(ews)
        ews = ew_fn(ews)
    else:
        cnt = None

    if not isinstance(eas, None.__class__):
        eas = ea_fn(eas)

    return TData(cur_slice, eis_t, x, ys, masks,
                 ews=ews, eas=eas, use_flows=use_flows,
                 cnt=cnt, node_map=nm)

def load_flows_pico(fname, start, end):
    """
    PicoDomain flows 슬라이스 파일 읽기
    형식: ts, src_id, dst_id, duration, orig_bytes, resp_bytes, orig_pkts, resp_pkts
    반환: {(src,dst): [num_flows, mean_dur, std_dur, mean_bytes, std_bytes, mean_pkts, std_pkts]}
    """
    eas_flows = {}
    temp = {}

    if not os.path.exists(fname):
        return eas_flows

    with open(fname, 'r') as f:
        for line in f:
            l = line.strip().split(',')
            if len(l) < 8:
                continue
            try:
                ts  = int(l[0])
                if ts < start:
                    continue
                if ts > end:
                    break
                src = int(l[1])
                dst = int(l[2])
                dur = float(l[3])
                ob  = float(l[4])
                rb  = float(l[5])
                op  = float(l[6])
                rp  = float(l[7])
                et  = (src, dst)
                if et not in temp:
                    temp[et] = [[], [], []]
                temp[et][0].append(dur)
                temp[et][1].append(ob + rb)
                temp[et][2].append(op + rp)
            except:
                continue

    for et, vals in temp.items():
        eas_flows[et] = [
            len(vals[0]),
            np.mean(vals[0]), np.std(vals[0]),
            np.mean(vals[1]), np.std(vals[1]),
            np.mean(vals[2]), np.std(vals[2]),
        ]
    return eas_flows

def load_partial_pico(start=0, end=65280, delta=3600,
                      is_test=False, use_flows=False,
                      ew_fn=standardized, ea_fn=std_edge_a):

    print(f'start:{start}, end:{end}')

    cur_slice = int(start - (start % FILE_DELTA))
    start_f   = str(cur_slice) + '.txt'

    # 파일 없는 구간 건너뛰기
    while not os.path.exists(PICO_FOLDER + start_f):
        cur_slice += FILE_DELTA
        start_f    = str(cur_slice) + '.txt'
        if cur_slice > end:
            return empty_pico(use_flows)

    in_f = open(PICO_FOLDER + start_f, 'r')

    edges     = []
    ews       = []
    edges_t   = {}
    ys        = []
    slices    = []
    eas       = []
    eas_flows = {}

    node_map = pickle.load(open(PICO_FOLDER + 'nmap.pkl', 'rb'))

    # 형식: ts, src_id, dst_id, label
    def fmt_line(x):
        x[-1] = x[-1].strip()
        return int(x[0]), int(x[1]), int(x[2]), int(x[3])

    def add_edge(et, is_anom=0):
        if et in edges_t:
            edges_t[et][0] = max(is_anom, edges_t[et][0])
            edges_t[et][1] += 1
        else:
            edges_t[et] = [is_anom, 1]

    scan_total = max(0, start - cur_slice - 1)
    scan_prog  = tqdm(desc='Finding start', total=scan_total)
    prog       = tqdm(desc='Seconds read',  total=end - start - 1)

    keep_reading = True
    next_split   = start + delta
    old_ts       = cur_slice
    curtime      = cur_slice

    if use_flows:
        eas_flows = load_flows_pico(
            PICO_FOLDER + 'flows/' + start_f, start, next_split)

    line = in_f.readline()

    while keep_reading:
        while line:
            l  = line.split(',')
            ts = int(l[0])

            if ts < start:
                scan_prog.update(max(0, ts - old_ts))
                old_ts  = ts
                curtime = ts
                line    = in_f.readline()
                continue

            try:
                ts, src, dst, label = fmt_line(l)
            except:
                line = in_f.readline()
                continue

            et = (src, dst)
            prog.update(max(0, ts - old_ts))
            old_ts = ts

            while ts >= next_split:
                if len(edges_t):
                    ei  = list(zip(*edges_t.keys()))
                    edges.append(ei)

                    vals = list(edges_t.values())
                    y   = [v[0] for v in vals]
                    ew  = [v[1] for v in vals]
                    ews.append(torch.tensor(ew))

                    if use_flows:
                        eas_flows_dim = 7
                        fs = {eij: (eas_flows[eij] if eij in eas_flows
                                    else [0.0] * eas_flows_dim)
                              for eij in edges_t.keys()}
                        ea_vals = list(zip(*fs.values()))
                        eas.append(torch.tensor(ea_vals))
                    else:
                        ew_feat = [[v[1]] for v in edges_t.values()]
                        eas.append(torch.tensor(ew_feat).transpose(1, 0))

                    if is_test:
                        ys.append(torch.tensor(y))

                    slices.append(str(next_split))
                    edges_t   = {}

                curtime    = next_split
                next_split += delta

                if use_flows:
                    flows_slice = int(curtime - (curtime % FILE_DELTA))
                    flows_file  = PICO_FOLDER + 'flows/' + str(flows_slice) + '.txt'
                    eas_flows   = load_flows_pico(flows_file, curtime, next_split)

                if curtime >= end:
                    keep_reading = False
                    break

            if not keep_reading:
                break

            if et[0] == et[1]:
                line = in_f.readline()
                continue

            add_edge(et, is_anom=label)
            line = in_f.readline()

        in_f.close()
        cur_slice += FILE_DELTA

        while cur_slice <= end:
            next_file = PICO_FOLDER + str(cur_slice) + '.txt'
            if os.path.exists(next_file):
                in_f = open(next_file, 'r')
                line = in_f.readline()
                break
            cur_slice += FILE_DELTA
        else:
            keep_reading = False
            break

    ys = ys if is_test else None
    scan_prog.close()
    prog.close()

    return make_data_obj(slices, edges, ys, ew_fn, ea_fn,
                         ews=ews, eas=eas, use_flows=use_flows,
                         node_map=node_map)
