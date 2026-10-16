from copy import deepcopy
import os
import pickle
import random

import torch
from tqdm import tqdm
import numpy as np

from .tdata import TData
from .load_utils import edge_tv_split, std_edge_w, standardized, std_edge_a

DATE_OF_EVIL_LANL = 150885
FILE_DELTA = 10000

# 가설 검증 실험용 설정 (main.py 옵션으로 바꿈)
FLOWS_COVERAGE = 1.0    # H1: 남길 플로우 레코드 비율
USE_AUTH_STATS = False  # H3: 플로우 특징 대신 인증 통계 사용
# 플로우 집계 방식
#   False: ARGUS 원래 방식. 10,000초 파일의 플로우 전체를 그 파일의 첫 스냅샷에 붙이고 나머지 스냅샷은 0벡터
#   True : 스냅샷마다 그 시간 안의 플로우만 집계
FLOWS_PER_SNAPSHOT = False

# Input the path where LANL data locates which should be the same as DST in split_lanl.py
LANL_FOLDER = './data/lanl/'

TIMES = {
    '5': 155399,
    '10': 210294,
    '20': 228642,
    '30': 464254,
    '40': 485925,
    '100': 740104,
    '500': 1089597,
    'all': 5011199,
    'test': 1270000,
}


def empty_lanl(use_flows=False):
    return make_data_obj(None, [], None, None, None, use_flows=use_flows)


def load_lanl_dist(start=0, end=635015, delta=8640, is_test=False, use_flows=False, ew_fn=std_edge_w, ea_fn=std_edge_a):
    if start is None or end is None:
        return empty_lanl(use_flows)
    return load_partial_lanl(start, end, delta, is_test, use_flows, ew_fn, ea_fn)


def make_data_obj(cur_slice, eis, ys, ew_fn, ea_fn, ews=None, eas=None, use_flows=False, **kwargs):
    if 'node_map' in kwargs:
        nm = kwargs['node_map']
    else:
        nm = pickle.load(open(LANL_FOLDER+'nmap.pkl', 'rb'))

    cl_cnt = len(nm)
    x = torch.eye(cl_cnt+1)
    # Build time-partitioned edge lists
    eis_t = []
    masks = []

    for i in range(len(eis)):
        ei = torch.tensor(eis[i])
        eis_t.append(ei)

        # This is training data if no ys present
        if isinstance(ys, None.__class__):
            masks.append(edge_tv_split(ei)[0])

    # Balance the edge weights if they exist
    if not isinstance(ews, None.__class__):
        cnt = deepcopy(ews)
        ews = ew_fn(ews)
    else:
        cnt = None

    if not isinstance(eas, None.__class__):
        eas = ea_fn(eas)

    return TData(cur_slice, eis_t, x, ys, masks, ews=ews, eas=eas, use_flows=use_flows, cnt=cnt, node_map=nm)


# 플로우 분할 파일을 엣지별 통계 7개로 집계 (ARGUS 원래 방식)
# [플로우 수, 지속시간 평균/표준편차, 패킷 수 평균/표준편차, 바이트 수 평균/표준편차]
def load_flows(fname, start, end, coverage=1.0):
    eas_flows = {}
    temp_flows = {}
    if not os.path.exists(fname):
        return eas_flows
    in_f = open(fname)
    line = in_f.readline()

    fmt_line = lambda x: (int(x[0]), int(x[1]), int(x[2]), int(x[6]), int(x[7]), int(x[8]))

    while line:
        l = line.split(',')
        ts = int(l[0])
        if ts < start:
            line = in_f.readline()
            continue
        if ts > end:
            break
        if random.random() > coverage:
            line = in_f.readline()
            continue
        ts, src, dst, duration, pck_cnt, byte_cnt = fmt_line(l)
        et = (src, dst)
        if et in temp_flows:
            temp_flows[et][0].append(duration)
            temp_flows[et][1].append(pck_cnt)
            temp_flows[et][2].append(byte_cnt)
        else:
            temp_flows[et] = [[duration], [pck_cnt], [byte_cnt]]
        line = in_f.readline()
    in_f.close()
    for et in temp_flows.keys():
        eas_flows[et] = [len(temp_flows[et][0]), np.mean(temp_flows[et][0]), np.std(temp_flows[et][0]),
                         np.mean(temp_flows[et][1]), np.std(temp_flows[et][1]),
                         np.mean(temp_flows[et][2]), np.std(temp_flows[et][2])]
    return eas_flows


# 플로우 분할 파일을 시간순 레코드로 읽음 (FLOWS_PER_SNAPSHOT=True): (ts, (src,dst), duration, pkt_cnt, byte_cnt)
def load_flows_records(fname, start, end, coverage=1.0):
    recs = []
    if not os.path.exists(fname):
        return recs
    with open(fname) as in_f:
        for line in in_f:
            l = line.split(',')
            ts = int(l[0])
            if ts < start:
                continue
            if ts > end:
                break
            if random.random() > coverage:
                continue
            recs.append((ts, (int(l[1]), int(l[2])), int(l[6]), int(l[7]), int(l[8])))
    return recs


# recs[lo:hi] 레코드를 엣지별 플로우 통계 7개로 집계 (FLOWS_PER_SNAPSHOT=True)
def aggregate_flows(recs, lo, hi):
    temp = {}
    for ts, et, d, p, b in recs[lo:hi]:
        if et in temp:
            temp[et][0].append(d)
            temp[et][1].append(p)
            temp[et][2].append(b)
        else:
            temp[et] = [[d], [p], [b]]
    out = {}
    for et, (d, p, b) in temp.items():
        out[et] = [len(d), np.mean(d), np.std(d), np.mean(p), np.std(p), np.mean(b), np.std(b)]
    return out


# H3 실험용. 플로우 특징 자리에 같은 엣지의 인증 통계 7개를 넣음
# edges_t_snapshot: {(src,dst): [label, count, uc, uu, ua]}
def load_auth_stats(edges_t_snapshot):
    auth_stats = {}
    for et, val in edges_t_snapshot.items():
        label, count, uc, uu, ua = val
        total = max(count, 1)
        auth_stats[et] = [
            float(count),
            float(uc) / total,
            float(uu) / total,
            float(ua) / total,
            float(uc + uu) / total,
            float(count) / max(total, 1),
            0.0,
        ]
    return auth_stats


def load_partial_lanl(start=140000, end=156659, delta=8640, is_test=False, use_flows=False, ew_fn=standardized, ea_fn=std_edge_a):
    print('start:' + str(start) + ', end:' + str(end))
    cur_slice = int(start - (start % FILE_DELTA))
    start_f = str(cur_slice) + '.txt'
    in_f = open(LANL_FOLDER + start_f, 'r')

    edges = []
    ews = []
    edges_t = {}
    ys = []
    slices = []
    eas = []
    eas_flows = {}

    # Predefined for easier loading so everyone agrees on NIDs
    node_map = pickle.load(open(LANL_FOLDER+'nmap.pkl', 'rb'))
    user_map = pickle.load(open(LANL_FOLDER+'umap.pkl', 'rb'))

    # line format ts,src,dst,src_u,dst_u,auth_t,logon_t,auth_o,success,label
    fmt_line = lambda x: (int(x[0]), int(x[1]), int(x[2]), int(x[9][:-1]), int(x[3]))

    # 출발 계정 종류: C(컴퓨터), U(사용자), 그 외
    def parse_user(src_u):
        if src_u[0] == 'C':
            return 2
        elif src_u[0] == 'U':
            return 3
        else:
            return 4

    def add_edge(et, src_u, is_anom=0):
        src_u_ind = parse_user(src_u)
        if et in edges_t:
            val = edges_t[et]
            edges_t[et][0:2] = [max(is_anom, val[0]), val[1]+1]
            edges_t[et][src_u_ind] = val[src_u_ind] + 1
        else:
            edges_t[et] = [0] * 5
            edges_t[et][0:2] = [is_anom, 1]
            edges_t[et][src_u_ind] = 1

    scan_prog = tqdm(desc='Finding start', total=start-cur_slice-1)
    prog = tqdm(desc='Seconds read', total=end-start-1)

    keep_reading = True
    next_split = start+delta

    line = in_f.readline()
    curtime = fmt_line(line.split(','))[0]
    old_ts = curtime

    # 플로우 레코드 버퍼와 현재 스냅샷 시작 시각
    flows_buf = []
    flows_idx = 0
    win_lo = start

    if use_flows:
        if not os.path.exists(LANL_FOLDER + '/flows'):
            print('flows has not been parsed')
        elif FLOWS_PER_SNAPSHOT:
            flows_buf = load_flows_records(LANL_FOLDER + '/flows/' + start_f, start, end, coverage=FLOWS_COVERAGE)
        else:
            eas_flows = load_flows(LANL_FOLDER + '/flows/' + start_f, start, end, coverage=FLOWS_COVERAGE)

    while keep_reading:
        while line:
            l = line.split(',')

            # Scan to the correct part of the file
            ts = int(l[0])
            if ts < start:
                line = in_f.readline()
                scan_prog.update(ts-old_ts)
                old_ts = ts
                curtime = ts
                continue

            ts, src, dst, label, src_u = fmt_line(l)
            et = (src, dst)
            src_u = user_map[src_u]

            prog.update(ts-old_ts)
            old_ts = ts

            # Split edge list if delta is hit
            if ts >= next_split:
                if len(edges_t):
                    ei = list(zip(*edges_t.keys()))
                    edges.append(ei)

                    # uc, uu, ua: 컴퓨터 계정, 사용자 계정, 그 외 인증 수
                    y, ew, uc, uu, ua = list(zip(*edges_t.values()))
                    ews.append(torch.tensor(ew))

                    if use_flows:
                        eas_flows_dim = 7
                        fs = {}

                        # 이 스냅샷 [win_lo, next_split) 안의 플로우만 집계
                        if FLOWS_PER_SNAPSHOT and not USE_AUTH_STATS:
                            while flows_idx < len(flows_buf) and flows_buf[flows_idx][0] < win_lo:
                                flows_idx += 1
                            hi = flows_idx
                            while hi < len(flows_buf) and flows_buf[hi][0] < next_split:
                                hi += 1
                            eas_flows = aggregate_flows(flows_buf, flows_idx, hi)
                            flows_idx = hi

                        if USE_AUTH_STATS:
                            auth_fs = load_auth_stats(edges_t)
                            for eij in edges_t.keys():
                                fs[eij] = auth_fs.get(eij, [0] * eas_flows_dim)
                        else:
                            # 플로우가 없는 엣지는 0벡터
                            for eij in edges_t.keys():
                                if eij in eas_flows:
                                    fs[eij] = eas_flows[eij]
                                else:
                                    fs[eij] = [0] * eas_flows_dim

                        num_flows, mean_duration, std_duration, mean_pkt_cnt, std_pkt_cnt, mean_byte_cnt, std_byte_cnt = list(zip(*fs.values()))
                        eas.append(torch.tensor([uc, uu, ua, num_flows, mean_duration, std_duration, mean_pkt_cnt, std_pkt_cnt, mean_byte_cnt, std_byte_cnt]))

                    else:
                        eas.append(torch.tensor([uc, uu, ua]))

                    if is_test:
                        ys.append(torch.tensor(y))

                    slices.append(str(ts))

                    edges_t = {}
                    eas_flows = {}

                # If the list was empty, just keep going if you can
                win_lo = next_split
                curtime = next_split
                next_split += delta

                # Break out of loop after saving if hit final timestep
                if ts >= end:
                    keep_reading = False
                    break

            # Skip self-loops
            if et[0] == et[1]:
                line = in_f.readline()
                continue

            add_edge(et, src_u, is_anom=label)
            line = in_f.readline()

        in_f.close()
        cur_slice += FILE_DELTA

        if os.path.exists(LANL_FOLDER + str(cur_slice) + '.txt'):
            in_f = open(LANL_FOLDER + str(cur_slice) + '.txt', 'r')
            line = in_f.readline()
            if use_flows and FLOWS_PER_SNAPSHOT:
                # 아직 안 쓴 플로우(파일 경계에 걸친 스냅샷 몫)는 남기고 다음 파일을 이어 붙임
                flows_buf = flows_buf[flows_idx:] + load_flows_records(
                    LANL_FOLDER + '/flows/' + str(cur_slice) + '.txt', start, end, coverage=FLOWS_COVERAGE)
                flows_idx = 0
            elif use_flows:
                eas_flows = load_flows(LANL_FOLDER + '/flows/' + str(cur_slice) + '.txt', start, end, coverage=FLOWS_COVERAGE)
        else:
            keep_reading = False
            break

    ys = ys if is_test else None

    scan_prog.close()
    prog.close()

    return make_data_obj(slices, edges, ys, ew_fn, ea_fn, ews=ews, eas=eas, use_flows=use_flows, node_map=node_map)
