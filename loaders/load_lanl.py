from copy import deepcopy
import os
import pickle
from joblib import Parallel, delayed

import torch
from torch_geometric.data import Data
from tqdm import tqdm

from .tdata import TData
from .load_utils import edge_tv_split, std_edge_w, standardized, std_edge_a

import random
import numpy as np

DATE_OF_EVIL_LANL = 150885
FLOWS_COVERAGE = 1  # E1 실험: 1.0=100%, 0.7=70%, 0.5=50%, 0.3=30%
USE_AUTH_STATS = True  # H3 실험: True로 설정하면 flows 대신 auth 통계 사용
FILE_DELTA = 10000

# Input the path where LANL data locates which should be the same as DST in split_lanl.py
LANL_FOLDER = 'C:/Users/user/Desktop/Argus/data/lanl/'
assert LANL_FOLDER, 'Please fill in the LANL_FOLDER in ./loaders/load_lanl.py'

TIMES = {
    '5': 155399,
    '10'      : 210294,
    '20'      : 228642,
    '30'      : 464254,
    '40'      : 485925,
    '100'     : 740104,
    '500'     : 1089597,
    'all'  : 5011199,
    'test' : 1270000,
    'gap7d' : 1270000,   # 테스트 시작: 훈련 끝 + 7일
    'gap14d': 1270000,   # 테스트 시작: 훈련 끝 + 14일
}

def empty_lanl(use_flows=False):
    return make_data_obj(None,[],None,None,None,use_flows=use_flows)

def load_lanl_dist(start=0, end=635015, delta=8640, is_test=False, use_flows=False, ew_fn=std_edge_w, ea_fn=std_edge_a):
    if start == None or end == None:
        return empty_lanl(use_flows)

    num_slices = ((end - start) // delta)
    remainder = (end-start) % delta
    num_slices = num_slices + 1 if remainder else num_slices
    return load_partial_lanl(start, end, delta, is_test, use_flows, ew_fn, ea_fn)

    per_worker = [num_slices // workers] * workers
    remainder = num_slices % workers

    if remainder:
        for i in range(workers, workers-remainder, -1):
            per_worker[i-1] += 1

    kwargs = []
    prev = start
    for i in range(workers):
        end_t = prev + delta*per_worker[i]
        kwargs.append({
            'start': prev,
            'end': min(end_t-1, end),
            'delta': delta,
            'is_test': is_test,
            'use_flows': use_flows,
            'ew_fn': ew_fn,
            'ea_fn': ea_fn
        })
        prev = end_t

    datas = Parallel(n_jobs=workers, prefer='processes')(delayed(load_partial_lanl_job)(i, kwargs[i]) for i in range(workers))

    # Helper method to concatonate one field from all of the datas
    data_reduce = lambda x : sum([getattr(datas[i], x) for i in range(workers)], [])

    # Just join all the lists from all the data objects
    print("Joining Data objects")
    slices = data_reduce('slices')
    x = datas[0].xs
    eis = data_reduce('eis')
    masks = data_reduce('masks')
    ews = data_reduce('ews')
    eas = data_reduce('eas')

    node_map = datas[0].node_map

    if is_test:
        ys = data_reduce('ys')
        cnt = data_reduce('cnt')
    else:
        ys = None
        cnt = None

    # After everything is combined, wrap it in a fancy new object, and you're
    # on your way to coolsville flats
    print("Done")
    return TData(slices, eis, x, ys, masks, ews=ews, eas=eas, node_map=node_map, cnt=cnt)

def load_partial_lanl_job(pid, args):
    data = load_partial_lanl(**args)
    return data

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

    #TODO: balance edge feature values
    if not isinstance(eas, None.__class__):
        eas = ea_fn(eas)

    return TData(cur_slice, eis_t, x, ys, masks, ews=ews, eas=eas, use_flows=use_flows, cnt=cnt, node_map=nm)


def load_flows(fname, start, end, coverage=1.0):
    eas_flows = {}
    temp_flows = {}
    if not os.path.exists(fname):
        return eas_flows
    in_f = open(fname)
    line = in_f.readline()

    fmt_line = lambda x : (int(x[0]), int(x[1]), int(x[2]), int(x[6]), int(x[7]), int(x[8]))

    while line:
        l = line.split(',')
        ts = int(l[0])
        if ts < start:
            line = in_f.readline()
            continue
        if ts > end:
            break
        # E1: coverage 비율로 무작위 샘플링
        if random.random() > coverage:
            line = in_f.readline()
            continue
        ts, src, dst, duration, pck_cnt, byte_cnt = fmt_line(l)
        et = (src,dst)
        if et in temp_flows:
            temp_flows[et][0].append(duration)
            temp_flows[et][1].append(pck_cnt)
            temp_flows[et][2].append(byte_cnt)
        else:
            temp_flows[et] = [[duration], [pck_cnt], [byte_cnt]]
        line = in_f.readline()
    in_f.close()
    #computes features, # of flows, mean & std of duration, pck_cnt and byte_cnt
    for et in temp_flows.keys():
        eas_flows[et] = [len(temp_flows[et][0]), np.mean(temp_flows[et][0]), np.std(temp_flows[et][0]), \
        np.mean(temp_flows[et][1]), np.std(temp_flows[et][1]), np.mean(temp_flows[et][2]), np.std(temp_flows[et][2])]
    return eas_flows

def load_auth_stats(edges_t_snapshot):
    """
    H3 실험: auth 이벤트에서 추출한 통계를 flows 자리에 사용
    edges_t_snapshot: {(src,dst): [label, count, uc, uu, ua]} 형태
    반환: {(src,dst): [7개 통계]} flows와 동일한 차원
    """
    auth_stats = {}
    for et, val in edges_t_snapshot.items():
        label, count, uc, uu, ua = val
        total = max(count, 1)
        # flows 7개 자리를 auth 통계로 대체
        auth_stats[et] = [
            float(count),           # num_events (flows의 num_flows 대체)
            float(uc) / total,      # C타입 user 비율 (mean_duration 대체)
            float(uu) / total,      # U타입 user 비율 (std_duration 대체)
            float(ua) / total,      # Anonymous user 비율 (mean_pkt_cnt 대체)
            float(uc + uu) / total, # 일반 user 비율 (std_pkt_cnt 대체)
            float(count) / max(total, 1),  # 정규화 이벤트 수 (mean_byte_cnt 대체)
            0.0,                    # 패딩 (std_byte_cnt 대체)
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


    # Helper functions (trims the trailing \n)
    # line format ts,src,dst,src_u,dst_u,auth_t,logon_t,auth_o,success,label
    fmt_line = lambda x : (int(x[0]), int(x[1]), int(x[2]), int(x[9][:-1]), int(x[3]))

    # take first char of src_u and convert it to edge list index
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

    anom_marked = False
    keep_reading = True
    next_split = start+delta

    line = in_f.readline()
    curtime = fmt_line(line.split(','))[0]
    old_ts = curtime

    #load flows if use_flows == True
    if use_flows:
        if not os.path.exists(LANL_FOLDER + '/flows'):
            print('flows has not been parsed')
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

            ts, src, dst, label, src_u= fmt_line(l)
            #Take the first char of src_u -> C, U or A, and the frequency of each type is the edge feature (3 features)
            et = (src,dst)
            src_u = user_map[src_u]

            # Not totally necessary but I like the loading bar
            prog.update(ts-old_ts)
            old_ts = ts

            # Split edge list if delta is hit
            if ts >= next_split:
                if len(edges_t):
                    ei = list(zip(*edges_t.keys()))
                    edges.append(ei)

                    #uc, us, ua: user C+, user U+, user Anonymous
                    y,ew,uc,uu,ua = list(zip(*edges_t.values()))
                    ews.append(torch.tensor(ew))

                    if use_flows:
                        eas_flows_dim = 7
                        fs = {}

                        if USE_AUTH_STATS:
                            # H3 실험: flows 대신 auth 통계 사용 (동종 소스)
                            auth_fs = load_auth_stats(edges_t)
                            for eij in edges_t.keys():
                                fs[eij] = auth_fs.get(eij, [0] * eas_flows_dim)
                        else:
                            # 정상 실험: 실제 flows 데이터 사용 (이종 소스)
                            for eij in edges_t.keys():
                                if eij in eas_flows:
                                    fs[eij] = eas_flows[eij]
                                else:
                                    fs[eij] = [0] * eas_flows_dim

                        num_flows,mean_duration,std_duration,mean_pkt_cnt,std_pkt_cnt,mean_byte_cnt,std_byte_cnt = list(zip(*fs.values()))
                        eas.append(torch.tensor([uc,uu,ua,num_flows,mean_duration,std_duration,mean_pkt_cnt,std_pkt_cnt,mean_byte_cnt,std_byte_cnt]))

                    else:
                        eas.append(torch.tensor([uc,uu,ua]))

                    if is_test:
                        ys.append(torch.tensor(y))

                    #a slice file might have multiple snapshots
                    #slices.append(str(cur_slice) + '.txt')
                    slices.append(str(ts))

                    edges_t = {}
                    eas_flows = {}

                # If the list was empty, just keep going if you can
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
            if use_flows:
                eas_flows = load_flows(LANL_FOLDER + '/flows/' + str(cur_slice) + '.txt', start, end, coverage=FLOWS_COVERAGE)
        else:
            keep_reading=False
            break

    ys = ys if is_test else None

    scan_prog.close()
    prog.close()


    return make_data_obj(slices, edges, ys, ew_fn, ea_fn, ews=ews, eas=eas, use_flows=use_flows, node_map=node_map)
