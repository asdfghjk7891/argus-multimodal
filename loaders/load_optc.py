from copy import deepcopy
import os
import pickle
from joblib import Parallel, delayed

import torch
from torch_geometric.data import Data
from tqdm import tqdm

from .tdata import TData
from .load_utils import edge_tv_split, std_edge_w, standardized, std_edge_a

import numpy as np

#DATE_OF_EVIL_LANL = 573290 #original 573290
DATE_OF_EVIL_LANL = 573290

TIMES = {
    '20'   : 573383,
    '100'  : 573751,
    '500'  : 575885,
    'all'  : 745983,
    'test' : 745983
}

FILE_DELTA = 10000

# Input the path where OpTC data locates which should be the same as DST in split_optc.py
OPTC_FOLDER = 'C:/Users/user/Desktop/Argus/data/'
assert OPTC_FOLDER, 'Please fill in the OPTC_FOLDER in loaders/load_optc.py'

# old_to_new 매핑 (기존 노드ID → optc_bro 노드ID)
def _build_old_to_new():
    new_nmap_path = OPTC_FOLDER + 'optc_bro/nmap_bro.pkl'
    if not os.path.exists(new_nmap_path):
        return {}
    old_nmap = pickle.load(open(OPTC_FOLDER + 'nmap.pkl', 'rb'))
    new_nmap = pickle.load(open(new_nmap_path, 'rb'))
    new_hostname_to_id = {v: i for i, v in enumerate(new_nmap) if v is not None}
    mapping = {}
    for old_id, val in enumerate(old_nmap):
        if val is None:
            continue
        try:
            num = int(val)
            hostname = f'SysClient{num:04d}.systemia.com'
            if hostname in new_hostname_to_id:
                mapping[old_id] = new_hostname_to_id[hostname]
        except:
            continue
    return mapping

old_to_new = _build_old_to_new()
assert OPTC_FOLDER, 'Please fill in the OPTC_FOLDER in loaders/load_optc.py'

TIMES = {
    '20'      : 573383, # First 20 anoms 1.55%
    '100'     : 573751, # First 100 anoms 11.7%
    '500'     : 575885, # First 500 anoms 18.73%
    'all'  : 745983,  # Full 21784
    'test' :  745983
}

def empty_lanl(use_flows=False):
    return make_data_obj(None,[],None,None,None,use_flows=use_flows)

def load_optc_dist(start=0, end=635015, delta=8640, is_test=False, use_flows=False, ew_fn=std_edge_w, ea_fn=std_edge_a):
    if start == None or end == None:
        return empty_lanl(use_flows)

    num_slices = ((end - start) // delta)
    remainder = (end-start) % delta
    num_slices = num_slices + 1 if remainder else num_slices
    # workers = min(num_slices, workers)

    # Can't distribute the job if not enough workers
    return load_partial_lanl(start, end, delta, is_test, use_flows, ew_fn, ea_fn)


# wrapper bc its annoying to send kwargs with Parallel
def load_partial_lanl_job(pid, args):
    data = load_partial_lanl(**args)
    return data


def make_data_obj(cur_slice, eis, ys, ew_fn, ea_fn, ews=None, eas=None, use_flows=False, **kwargs):
    if 'node_map' in kwargs:
        nm = kwargs['node_map']
    else:
        nm = pickle.load(open(OPTC_FOLDER+'nmap.pkl', 'rb'))

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
    # print(eas)

    #TODO: balance edge feature values
    # if not isinstance(eas, None.__class__):
    #     eas = ea_fn(eas)
    # exit()
    # Finally, return Data object
    return TData(
        cur_slice, eis_t, x, ys, masks, ews=ews, eas=eas, use_flows=use_flows, cnt=cnt, node_map=nm
    )

'''
Read a file in flows and return the edge features

'''
def load_flows(fname, start, end):
    #TODO: implement
    eas_flows = {}
    temp_flows = {}
    if not os.path.exists(fname):
        return eas_flows
    in_f = open(fname)
    line = in_f.readline()

    #Line in parsed flows. ts, src, dst,src_port,dst_port,proto, duration, pck_cnt, byte_cnt, label
    fmt_line = lambda x : (int(x[0]), int(x[1]), int(x[2]), int(x[6]), int(x[7]), int(x[8]))

    while line:
        l = line.split(',')
        ts = int(l[0])
        if ts < start:
            line = in_f.readline()
            continue
        if ts > end:
            break
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

'''
Equivilant to load_cyber.load_lanl but uses the sliced LANL files
for faster scanning to the correct lines
'''
def load_partial_lanl(start=140000, end=156659, delta=8640, is_test=False, use_flows=False, ew_fn=standardized, ea_fn=std_edge_a):
    print('start:' + str(start) + ', end:' + str(end))
    cur_slice = int(start - (start % FILE_DELTA))
    start_f = str(cur_slice) + '.txt'
    in_f = open(OPTC_FOLDER + start_f, 'r')

    edges = []
    ews = []
    edges_t = {}
    ys = []
    slices = []
    eas = []

    # E4: Gaussian 샘플링 캐시 (pkl로 저장/로드)
    flows_stats_cache = {}
    if use_flows:
        cache_path = OPTC_FOLDER + 'optc_bro/flows_stats_cache.pkl'
        if os.path.exists(cache_path):
            flows_stats_cache = pickle.load(open(cache_path, 'rb'))
            print(f'[E4] flows 캐시 로드 완료: {len(flows_stats_cache)}개 엣지')
        else:
            flows_dir = OPTC_FOLDER + 'optc_bro/flows/'
            temp_cache = {}
            for fname in sorted(os.listdir(flows_dir)):
                if not fname.endswith('.txt'):
                    continue
                fpath = os.path.join(flows_dir, fname)
                if os.path.getsize(fpath) == 0:
                    continue
                with open(fpath, 'r') as ff:
                    for fline in ff:
                        fl = fline.strip().split(',')
                        if len(fl) < 8:
                            continue
                        try:
                            et = (int(fl[1]), int(fl[2]))
                            dur = float(fl[3])
                            ob  = int(fl[4]) + int(fl[5])
                            op  = int(fl[6]) + int(fl[7])
                            if et not in temp_cache:
                                temp_cache[et] = [[], [], []]
                            temp_cache[et][0].append(dur)
                            temp_cache[et][1].append(ob)
                            temp_cache[et][2].append(op)
                        except:
                            continue
            for et, v in temp_cache.items():
                flows_stats_cache[et] = [
                    np.mean(v[0]), max(np.std(v[0]), 1e-6),
                    np.mean(v[1]), max(np.std(v[1]), 1e-6),
                    np.mean(v[2]), max(np.std(v[2]), 1e-6),
                ]
            pickle.dump(flows_stats_cache, open(cache_path, 'wb'),
                       protocol=pickle.HIGHEST_PROTOCOL)
            print(f'[E4] flows 캐시 구축 및 저장 완료: {len(flows_stats_cache)}개 엣지')
            del temp_cache

    # Predefined for easier loading so everyone agrees on NIDs
    node_map = pickle.load(open(OPTC_FOLDER+'nmap.pkl', 'rb'))

    # Helper functions (trims the trailing \n)
    #ZL: line format ts,src,dst,src_u,dst_u,auth_t,logon_t,auth_o,success,label
    # fmt_line = lambda x : (int(x[0]), int(x[1]), int(x[2]), int(x[9][:-1]), int(x[3]))
    fmt_line = lambda x : (int(x[0]), int(x[1]), int(x[2]), int(x[3]), int(x[4]), int(x[5]), int(x[6]), int(x[7]),int(x[8][:-1]))

    # take first char of src_u and convert it to edge list index
    # def parse_user(src_u):
    #     if src_u[0] == 'C':
    #         return 2
    #     elif src_u[0] == 'U':
    #         return 3
    #     else:
    #         return 4

    # For now, just keeps one copy of each edge. Could be
    # modified in the future to add edge weight or something
    # but for now, edges map to their anomaly value (1 == anom, else 0)
    # TODO: include edge features
    def add_edge(et, ea, is_anom=0):
        pid, ppid, dest_port, l4protocol, img_path = ea
        if et in edges_t:
            val = edges_t[et]
            edges_t[et][0] = max(is_anom, val[0])
            edges_t[et][1] += 1
            edges_t[et][2].add(dest_port)
            edges_t[et][3].add(l4protocol)
            edges_t[et][4].add(img_path)
            edges_t[et][5].add(pid)
        else:
            edges_t[et] = [
                is_anom,            # 공격 여부
                1,                  # 연결 횟수
                {dest_port},        # 고유 port 집합
                {l4protocol},       # 고유 protocol 집합
                {img_path},         # 고유 img_path 집합
                {pid},              # 고유 pid 집합
            ]

    # def add_edge(et, src_u, is_anom=0):
    #     src_u_ind = parse_user(src_u)
    #     if et in edges_t:
    #         val = edges_t[et]
    #         edges_t[et][0:2] = [max(is_anom, val[0]), val[1]+1]
    #         edges_t[et][src_u_ind] = val[src_u_ind] + 1
    #     else:
    #         edges_t[et] = [0] * 5
    #         edges_t[et][0:2] = [is_anom, 1]
    #         edges_t[et][src_u_ind] = 1



    scan_prog = tqdm(desc='Finding start', total=start-cur_slice-1)
    prog = tqdm(desc='Seconds read', total=end-start-1)

    anom_marked = False
    keep_reading = True
    next_split = start+delta

    line = in_f.readline()
    curtime = fmt_line(line.split(','))[0]
    old_ts = curtime

    #load flows if use_flows == True
    # if use_flows:
    #     if not os.path.exists(OPTC_FOLDER + '/flows'):
    #         print('flows has not been parsed')
    #     else:
    #         eas_flows = load_flows(OPTC_FOLDER + '/flows/' + start_f, start, end)

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

            # ['timestamps', 'source', 'target', 'pid', 'ppid', 'dest_port', 'l4protocol', 'img_path', 'label']
            # ts, src, dst, label, src_u= fmt_line(l)
            ts, src, dst, pid, ppid, dest_port, l4protocol, img_path, label = fmt_line(l)
            ea = (int(pid), int(ppid), int(dest_port), int(l4protocol), int(img_path))
            # eas.append(torch.tensor([pid, ppid, dest_port, l4protocol, img_path]))

            #Take the first char of src_u -> C, U or A, and the frequency of each type is the edge feature (3 features)
            et = (src,dst)
            # src_u = user_map[src_u]

            # Not totally necessary but I like the loading bar
            prog.update(ts-old_ts)
            old_ts = ts

            # Split edge list if delta is hit
            if ts >= next_split:
                if len(edges_t):
                    ei = list(zip(*edges_t.keys()))
                    edges.append(ei)

                    # edges_t 값 추출
                    vals = list(edges_t.values())
                    y  = [v[0] for v in vals]
                    ew = [v[1] for v in vals]
                    ews.append(torch.tensor(ew))

                    if use_flows:
                        # optc_bro/flows/ 에서 bro 통계 조회
                        flows_slice = int(curtime - (curtime % 10000))
                        flows_file = OPTC_FOLDER + 'optc_bro/flows/' + str(flows_slice) + '.txt'

                        bro_flows = {}
                        if os.path.exists(flows_file):
                            with open(flows_file, 'r') as ff:
                                for fline in ff:
                                    fl = fline.strip().split(',')
                                    if len(fl) < 8:
                                        continue
                                    try:
                                        fn_src = int(fl[1])
                                        fn_dst = int(fl[2])
                                        dur = float(fl[3])
                                        ob  = int(fl[4])
                                        rb  = int(fl[5])
                                        op  = int(fl[6])
                                        rp  = int(fl[7])
                                        et_new = (fn_src, fn_dst)
                                        if et_new not in bro_flows:
                                            bro_flows[et_new] = [[], [], []]
                                        bro_flows[et_new][0].append(dur)
                                        bro_flows[et_new][1].append(ob + rb)
                                        bro_flows[et_new][2].append(op + rp)
                                    except:
                                        continue
                        ea_list = []
                        for et_key in edges_t.keys():
                            if et_key in bro_flows:
                                # 실제 flows 데이터 사용
                                v = bro_flows[et_key]
                                ea_list.append([
                                    len(v[0]),
                                    np.mean(v[0]), np.std(v[0]),
                                    np.mean(v[1]), np.std(v[1]),
                                    np.mean(v[2]), np.std(v[2]),
                                ])
                            elif et_key in flows_stats_cache:
                                # Gaussian 샘플링: 과거 통계에서 샘플링
                                mu_d, si_d, mu_b, si_b, mu_p, si_p = flows_stats_cache[et_key]
                                sampled_dur   = max(0.0, np.random.normal(mu_d, si_d))
                                sampled_bytes = max(0.0, np.random.normal(mu_b, si_b))
                                sampled_pkts  = max(0.0, np.random.normal(mu_p, si_p))
                                ea_list.append([
                                    1.0,           # num_flows=1 (시뮬레이션)
                                    sampled_dur,   0.0,
                                    sampled_bytes, 0.0,
                                    sampled_pkts,  0.0,
                                ])
                            else:
                                ea_list.append([0.0] * 7)

                        ea_tensor = torch.tensor(ea_list, dtype=torch.float32)
                        ea_tensor = torch.log1p(ea_tensor)
                        eas.append(ea_tensor.transpose(1, 0))  # (7, num_edges)
                        if len(eas) == 1:
                            matched_real = sum(1 for et_key in edges_t.keys() if et_key in bro_flows)
                            matched_sim  = sum(1 for et_key in edges_t.keys() if et_key not in bro_flows and et_key in flows_stats_cache)
                            print(f"[DEBUG] eas[0] shape: {eas[0].shape}")
                            print(f"[DEBUG] eas[0] sample:\n{eas[0][:, :3]}")
                            print(f"[DEBUG] 실제 flows: {matched_real}, Gaussian 시뮬: {matched_sim}, 0벡터: {len(ea_list)-matched_real-matched_sim}")

                    if is_test:
                        ys.append(torch.tensor(y))

                    #a slice file might have multiple snapshots
                    #slices.append(str(cur_slice) + '.txt')
                    slices.append(str(ts))

                    edges_t = {}

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
            add_edge(et, ea, is_anom=label)
            # add_edge(et, src_u, is_anom=label)
            line = in_f.readline()

        in_f.close()
        cur_slice += FILE_DELTA

        if os.path.exists(OPTC_FOLDER + str(cur_slice) + '.txt'):
            in_f = open(OPTC_FOLDER + str(cur_slice) + '.txt', 'r')
            line = in_f.readline()
        else:
            keep_reading=False
            break

    ys = ys if is_test else None

    scan_prog.close()
    prog.close()
    return make_data_obj(
        slices, edges, ys, ew_fn, ea_fn,
        ews=ews, eas=eas, use_flows=use_flows, node_map=node_map
    )
