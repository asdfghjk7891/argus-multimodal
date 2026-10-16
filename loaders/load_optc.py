from copy import deepcopy
import os
import pickle

import torch
from tqdm import tqdm
import numpy as np

from .tdata import TData
from .load_utils import edge_tv_split, std_edge_w, standardized, std_edge_a

DATE_OF_EVIL_LANL = 573290
FILE_DELTA = 10000

TIMES = {
    '20': 573383,    # First 20 anoms 1.55%
    '100': 573751,   # First 100 anoms 11.7%
    '500': 575885,   # First 500 anoms 18.73%
    'all': 745983,   # Full 21784
    'test': 745983
}

# Input the path where OpTC data locates (ARGUS OpTC 전처리 결과 폴더)
OPTC_FOLDER = './data/'


# 원본 그래프 노드 ID -> ecar-bro 노드 ID (hostname 기준, split_optc_bro.py 결과 필요)
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
        except Exception:
            continue
    return mapping


old_to_new = _build_old_to_new()


def empty_lanl(use_flows=False):
    return make_data_obj(None, [], None, None, None, use_flows=use_flows)


def load_optc_dist(start=0, end=635015, delta=8640, is_test=False, use_flows=False, ew_fn=std_edge_w, ea_fn=std_edge_a):
    if start is None or end is None:
        return empty_lanl(use_flows)
    return load_partial_lanl(start, end, delta, is_test, use_flows, ew_fn, ea_fn)


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

    return TData(
        cur_slice, eis_t, x, ys, masks, ews=ews, eas=eas, use_flows=use_flows, cnt=cnt, node_map=nm
    )


# optc_bro/flows/<slice>.txt 를 읽어 엣지별 플로우 레코드를 모음
def load_bro_flows(flows_file):
    bro_flows = {}
    if not os.path.exists(flows_file):
        return bro_flows
    with open(flows_file, 'r') as ff:
        for fline in ff:
            fl = fline.strip().split(',')
            if len(fl) < 8:
                continue
            try:
                et_new = (int(fl[1]), int(fl[2]))
                dur = float(fl[3])
                ob, rb = int(fl[4]), int(fl[5])
                op, rp = int(fl[6]), int(fl[7])
                if et_new not in bro_flows:
                    bro_flows[et_new] = [[], [], []]
                bro_flows[et_new][0].append(dur)
                bro_flows[et_new][1].append(ob + rb)
                bro_flows[et_new][2].append(op + rp)
            except Exception:
                continue
    return bro_flows


def load_partial_lanl(start=140000, end=156659, delta=8640, is_test=False, use_flows=False, ew_fn=standardized, ea_fn=std_edge_a):
    global old_to_new
    if use_flows and not old_to_new:
        old_to_new = _build_old_to_new()  # --data_path로 경로가 바뀐 경우
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

    # Predefined for easier loading so everyone agrees on NIDs
    node_map = pickle.load(open(OPTC_FOLDER+'nmap.pkl', 'rb'))

    # line format: ts, src, dst, pid, ppid, dest_port, l4protocol, img_path, label
    fmt_line = lambda x: (int(x[0]), int(x[1]), int(x[2]), int(x[3]), int(x[4]), int(x[5]), int(x[6]), int(x[7]), int(x[8][:-1]))

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
            edges_t[et] = [is_anom, 1, {dest_port}, {l4protocol}, {img_path}, {pid}]

    scan_prog = tqdm(desc='Finding start', total=start-cur_slice-1)
    prog = tqdm(desc='Seconds read', total=end-start-1)

    keep_reading = True
    next_split = start+delta

    line = in_f.readline()
    curtime = fmt_line(line.split(','))[0]
    old_ts = curtime

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

            ts, src, dst, pid, ppid, dest_port, l4protocol, img_path, label = fmt_line(l)
            ea = (int(pid), int(ppid), int(dest_port), int(l4protocol), int(img_path))
            et = (src, dst)

            prog.update(ts-old_ts)
            old_ts = ts

            # Split edge list if delta is hit
            if ts >= next_split:
                if len(edges_t):
                    ei = list(zip(*edges_t.keys()))
                    edges.append(ei)

                    vals = list(edges_t.values())
                    y = [v[0] for v in vals]
                    ew = [v[1] for v in vals]
                    ews.append(torch.tensor(ew))

                    if use_flows:
                        # 원본 그래프 엣지에 hostname이 같은 bro 플로우 통계를 붙임. 없으면 0벡터
                        flows_slice = int(curtime - (curtime % 10000))
                        bro_flows = load_bro_flows(OPTC_FOLDER + 'optc_bro/flows/' + str(flows_slice) + '.txt')

                        ea_list = []
                        for et_key in edges_t.keys():
                            et_bro = (old_to_new.get(et_key[0], -1), old_to_new.get(et_key[1], -1))
                            if et_bro[0] != -1 and et_bro[1] != -1 and et_bro in bro_flows:
                                v = bro_flows[et_bro]
                                ea_list.append([
                                    len(v[0]),
                                    np.mean(v[0]), np.std(v[0]),
                                    np.mean(v[1]), np.std(v[1]),
                                    np.mean(v[2]), np.std(v[2]),
                                ])
                            else:
                                ea_list.append([0.0] * 7)

                        ea_tensor = torch.tensor(ea_list, dtype=torch.float32)
                        ea_tensor = torch.log1p(ea_tensor)
                        eas.append(ea_tensor.transpose(1, 0))  # (7, num_edges)

                    if is_test:
                        ys.append(torch.tensor(y))

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
            line = in_f.readline()

        in_f.close()
        cur_slice += FILE_DELTA

        if os.path.exists(OPTC_FOLDER + str(cur_slice) + '.txt'):
            in_f = open(OPTC_FOLDER + str(cur_slice) + '.txt', 'r')
            line = in_f.readline()
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
