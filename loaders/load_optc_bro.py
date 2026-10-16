from copy import deepcopy
import os
import pickle

import torch
from tqdm import tqdm
import numpy as np

from .tdata import TData
from .load_utils import edge_tv_split, std_edge_w, standardized, std_edge_a

# OpTC 초기 실험용 로더 (main.py --bro).
# flows 미사용: 원본 OpTC 그래프(OPTC_FOLDER), flows 사용: ecar-bro 그래프(OPTC_BRO_FOLDER)

DATE_OF_EVIL_LANL = 560000
FILE_DELTA = 10000

OPTC_FOLDER = './data/'
OPTC_BRO_FOLDER = './data/optc_bro/'

TIMES = {
    '20': 573383,
    '100': 573751,
    '500': 575885,
    'all': 745983,
    'test': 745983
}


def empty_lanl(use_flows=False):
    return make_data_obj(None, [], None, None, None, use_flows=use_flows)


def load_optc_dist(start=0, end=635015, delta=8640, is_test=False,
                   use_flows=False, ew_fn=std_edge_w, ea_fn=std_edge_a):
    if start is None or end is None:
        return empty_lanl(use_flows)
    return load_partial_lanl(start, end, delta, is_test, use_flows, ew_fn, ea_fn)


def make_data_obj(cur_slice, eis, ys, ew_fn, ea_fn,
                  ews=None, eas=None, use_flows=False, **kwargs):
    if 'node_map' in kwargs:
        nm = kwargs['node_map']
    else:
        if use_flows and os.path.exists(OPTC_BRO_FOLDER + 'nmap_bro.pkl'):
            nm = pickle.load(open(OPTC_BRO_FOLDER + 'nmap_bro.pkl', 'rb'))
        else:
            nm = pickle.load(open(OPTC_FOLDER + 'nmap.pkl', 'rb'))

    cl_cnt = len(nm)
    x = torch.eye(cl_cnt + 1)

    eis_t = []
    masks = []

    for i in range(len(eis)):
        ei = torch.tensor(eis[i])
        eis_t.append(ei)
        if isinstance(ys, None.__class__):
            masks.append(edge_tv_split(ei)[0])

    if not isinstance(ews, None.__class__):
        cnt = deepcopy(ews)
        ews = ew_fn(ews)
    else:
        cnt = None

    return TData(
        cur_slice, eis_t, x, ys, masks,
        ews=ews, eas=eas, use_flows=use_flows, cnt=cnt, node_map=nm
    )


# bro conn 기반 플로우 슬라이스 파일을 엣지별 통계 7개로 집계
# 파일 형식: ts, src, dst, duration, orig_bytes, resp_bytes, orig_pkts, resp_pkts, dest_port
def load_flows_bro(fname):
    eas_flows = {}
    temp_flows = {}

    if not os.path.exists(fname):
        return eas_flows

    with open(fname, 'r') as f:
        for line in f:
            l = line.strip().split(',')
            if len(l) < 8:
                continue
            try:
                src = int(l[1])
                dst = int(l[2])
                duration = float(l[3])
                orig_b = int(l[4])
                resp_b = int(l[5])
                orig_p = int(l[6])
                resp_p = int(l[7])

                et = (src, dst)
                total_bytes = orig_b + resp_b
                total_pkts = orig_p + resp_p

                if et in temp_flows:
                    temp_flows[et][0].append(duration)
                    temp_flows[et][1].append(total_bytes)
                    temp_flows[et][2].append(total_pkts)
                else:
                    temp_flows[et] = [[duration], [total_bytes], [total_pkts]]
            except Exception:
                continue

    for et, vals in temp_flows.items():
        dur_list, bytes_list, pkts_list = vals
        eas_flows[et] = [
            len(dur_list),
            np.mean(dur_list), np.std(dur_list),
            np.mean(bytes_list), np.std(bytes_list),
            np.mean(pkts_list), np.std(pkts_list),
        ]

    return eas_flows


def load_partial_lanl(start=140000, end=156659, delta=8640,
                      is_test=False, use_flows=False,
                      ew_fn=standardized, ea_fn=std_edge_a):

    DATA_FOLDER = OPTC_BRO_FOLDER if use_flows else OPTC_FOLDER
    NMAP_FILE = 'nmap_bro.pkl' if use_flows else 'nmap.pkl'

    print('start:' + str(start) + ', end:' + str(end))

    cur_slice = int(start - (start % FILE_DELTA))
    start_f = str(cur_slice) + '.txt'
    in_f = open(DATA_FOLDER + start_f, 'r')

    edges = []
    ews = []
    edges_t = {}
    ys = []
    slices = []
    eas = []

    node_map = pickle.load(open(DATA_FOLDER + NMAP_FILE, 'rb'))

    # 플로우 슬라이스 캐시 (파일당 한 번만 읽음)
    flows_cache = {}

    def get_flows(slice_ts):
        if slice_ts not in flows_cache:
            fname = OPTC_BRO_FOLDER + 'flows/' + str(slice_ts) + '.txt'
            flows_cache[slice_ts] = load_flows_bro(fname)
        return flows_cache[slice_ts]

    # ecar-bro: ts, src, dst, label / 원본: ts, src, dst, pid, ppid, dest_port, l4protocol, img_path, label
    if use_flows:
        def fmt_line(x):
            x[-1] = x[-1].strip()
            return (int(x[0]), int(x[1]), int(x[2]), int(x[3]))
    else:
        def fmt_line(x):
            x[-1] = x[-1].strip()
            return (int(x[0]), int(x[1]), int(x[2]),
                    int(x[3]), int(x[4]), int(x[5]),
                    int(x[6]), int(x[7]), int(x[8]))

    def add_edge(et, is_anom=0):
        if et in edges_t:
            edges_t[et][0] = max(is_anom, edges_t[et][0])
            edges_t[et][1] += 1
        else:
            edges_t[et] = [is_anom, 1]

    scan_total = max(0, start - cur_slice - 1)
    scan_prog = tqdm(desc='Finding start', total=scan_total)
    prog = tqdm(desc='Seconds read', total=end - start - 1)

    keep_reading = True
    next_split = start + delta
    curtime = cur_slice
    old_ts = cur_slice

    line = in_f.readline()

    while keep_reading:
        while line:
            l = line.split(',')
            ts = int(l[0])

            if ts < start:
                scan_prog.update(max(0, ts - old_ts))
                old_ts = ts
                curtime = ts
                line = in_f.readline()
                continue

            try:
                if use_flows:
                    ts, src, dst, label = fmt_line(l)
                else:
                    ts, src, dst, pid, ppid, dp, l4p, ip, label = fmt_line(l)
            except Exception:
                line = in_f.readline()
                continue

            et = (src, dst)

            prog.update(max(0, ts - old_ts))
            old_ts = ts

            # ts가 여러 스냅샷을 건너뛰는 경우도 처리
            while ts >= next_split:
                if len(edges_t):
                    ei = list(zip(*edges_t.keys()))
                    edges.append(ei)

                    vals = list(edges_t.values())
                    y = [v[0] for v in vals]
                    ew = [v[1] for v in vals]
                    ews.append(torch.tensor(ew))

                    if use_flows:
                        flows_slice = int(curtime - (curtime % FILE_DELTA))
                        eas_flows = get_flows(flows_slice)

                        ea_list = []
                        for et_key in edges_t.keys():
                            if et_key in eas_flows:
                                ea_list.append(eas_flows[et_key])
                            else:
                                ea_list.append([0.0]*7)

                        ea_tensor = torch.tensor(ea_list, dtype=torch.float32)
                        ea_tensor = torch.log1p(ea_tensor)
                        eas.append(ea_tensor.transpose(1, 0))

                    if is_test:
                        ys.append(torch.tensor(y))

                    slices.append(str(next_split))
                    edges_t = {}

                curtime = next_split
                next_split += delta

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

        # 파일이 없는 구간은 건너뜀
        while cur_slice <= end:
            next_file = DATA_FOLDER + str(cur_slice) + '.txt'
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

    return make_data_obj(
        slices, edges, ys, ew_fn, ea_fn,
        ews=ews, eas=eas, use_flows=use_flows, node_map=node_map
    )
