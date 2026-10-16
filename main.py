from argparse import ArgumentParser
import os
import datetime
import random

import numpy as np
import torch

import loaders.load_optc as optc
import loaders.load_optc_bro as optc_bro
import loaders.load_lanl as lanl
import loaders.load_cert as cert
import loaders.load_picodomain as pico
from models.recurrent import GRU, LSTM, EmptyModel
from models.argus import (detector_lanl_rref, detector_lanl_late_rref, detector_lanl_uniflows_rref,
                          detector_optc_rref, detector_optc_late_rref, detector_optc_earlyfusion_rref,
                          detector_cert_rref, detector_pico_late_rref)
from classification import classification


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"seed = {seed}")


def args():
    ap = ArgumentParser()
    ap.add_argument('-d', '--delta', type=float, default=1)
    ap.add_argument('-e', '--encoder_name', type=str.upper, default="ARGUS")
    ap.add_argument('-r', '--rnn', choices=['GRU', 'LSTM', 'NONE'], type=str.upper, default="GRU")
    ap.add_argument('-H', '--hidden', type=int, default=32)
    ap.add_argument('-z', '--zdim', type=int, default=16)
    ap.add_argument('-l', '--load', action='store_true')
    ap.add_argument('--gpu', action='store_true')
    # The end of testing time, see load_lanl.TIMES
    ap.add_argument('-te', '--te_end', choices=['20', '100', '500', 'all', 'test'], type=str.lower, default="test")
    ap.add_argument('--fpweight', type=float, default=0.6)
    ap.add_argument('--dataset', default='LANL', type=str.upper, choices=['OPTC', 'LANL', 'CERT', 'PICO'])
    ap.add_argument('--lr', default=0.01, type=float)
    ap.add_argument('--patience', default=3, type=int)
    ap.add_argument('--nratio', default=1, type=int)
    ap.add_argument('--epochs', default=100, type=int)
    ap.add_argument('--loss', type=str, default="default", choices=['default', 'ap', 'bce'])
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--data_path', type=str, default=None,
                    help='전처리 데이터 폴더. 지정하지 않으면 각 로더의 기본 경로 사용')

    # 멀티모달 옵션
    ap.add_argument('--flows', action='store_true', help='보조 모달리티(플로우) 사용')
    ap.add_argument('--fusion', type=str, default='early', choices=['early', 'late', 'uniflows'],
                    help='early: Early Fusion, late: Late Fusion(MAGNET), uniflows: 플로우만 사용')
    ap.add_argument('--bro', action='store_true',
                    help='OpTC 초기 실험 설정(bro 기록 구간에 맞춘 훈련/테스트 구간)')

    # LANL 가설 검증 실험(H1~H3)
    ap.add_argument('--coverage', type=float, default=1.0, help='H1: 남길 플로우 레코드 비율')
    ap.add_argument('--gap', type=int, default=0, help='H2: 훈련 끝과 테스트 시작 사이 공백(초)')
    ap.add_argument('--auth_stats', action='store_true', help='H3: 플로우 특징 대신 인증 통계 사용')
    ap.add_argument('--flows_per_snapshot', action='store_true',
                    help='LANL 플로우를 스냅샷마다 집계 (기본은 ARGUS 원래 방식)')

    args = ap.parse_args()
    assert 0 <= args.fpweight <= 1, '--fpweight must be a value between 0 and 1 (inclusive)'

    set_seed(args.seed)

    readable = str(args)
    print(readable)
    model_str = '%s -> %s ' % (args.encoder_name, args.rnn)
    print(model_str)
    args.dataset = args.dataset + '_' + args.encoder_name

    if args.data_path is not None:
        args.data_path = args.data_path.replace('\\', '/')
        if not args.data_path.endswith('/'):
            args.data_path += '/'

    # Parse dataset info
    if args.dataset.startswith('O'):
        if args.bro:
            if args.data_path is not None:
                optc_bro.OPTC_FOLDER = args.data_path
                optc_bro.OPTC_BRO_FOLDER = args.data_path + 'optc_bro/'
            args.loader = optc_bro.load_optc_dist
            args.tr_start = 0
            args.tr_end = 76000
            args.val_times = (76000, 80000)
            args.te_times = [(560000, 690000)]
        else:
            if args.data_path is not None:
                optc.OPTC_FOLDER = args.data_path
            args.loader = optc.load_optc_dist
            args.tr_start = 0
            args.tr_end = optc.DATE_OF_EVIL_LANL
            args.val_times = None
            args.te_times = [(args.tr_end, optc.TIMES[args.te_end])]
        args.delta = int(args.delta * (60**2))
    elif args.dataset.startswith('L'):
        if args.data_path is not None:
            lanl.LANL_FOLDER = args.data_path
        lanl.FLOWS_COVERAGE = args.coverage
        lanl.USE_AUTH_STATS = args.auth_stats
        lanl.FLOWS_PER_SNAPSHOT = args.flows_per_snapshot
        args.loader = lanl.load_lanl_dist
        args.tr_start = 0
        args.tr_end = lanl.DATE_OF_EVIL_LANL
        args.val_times = None
        args.te_times = [(args.tr_end + args.gap, lanl.TIMES[args.te_end])]
        args.delta = int(args.delta * (60**2))
    elif args.dataset.startswith('C'):
        if args.data_path is not None:
            cert.CERT_FOLDER = args.data_path + 'cert/'
        meta = cert.load_meta()
        if meta:
            cert.DATE_OF_EVIL_CERT = meta['DATE_OF_EVIL']
            cert.TIMES['test'] = meta['total_duration']
        args.loader = cert.load_cert_dist
        args.tr_start = 0
        args.tr_end = cert.DATE_OF_EVIL_CERT
        args.val_times = None
        args.te_times = [(args.tr_end, cert.TIMES['test'])]
        args.delta = int(args.delta * 86400)  # 1일 단위
    elif args.dataset.startswith('P'):
        # PicoDomain은 AP loss 설정이 없어서 bce로 고정
        if args.loss != 'bce':
            print(f"PICO는 AP loss를 지원하지 않아 --loss {args.loss} 대신 bce 사용")
            args.loss = 'bce'
        if args.data_path is not None:
            pico.PICO_FOLDER = args.data_path + 'picodomain/'
        meta = pico.load_meta()
        if meta:
            pico.DATE_OF_EVIL_PICO = meta['DATE_OF_EVIL']
            pico.TIMES['test'] = meta['total_duration']
        args.loader = pico.load_pico_dist
        args.tr_start = 0
        _delta = int(args.delta * 3600)
        # 공격 시작 직전 2개 스냅샷을 검증 구간으로 사용
        val_end = (pico.DATE_OF_EVIL_PICO // _delta) * _delta
        val_start = val_end - _delta * 2
        args.val_times = (val_start, val_end)
        args.tr_end = val_start
        args.te_times = [(pico.DATE_OF_EVIL_PICO, pico.TIMES['test'])]
        args.delta = _delta
    else:
        raise NotImplementedError('Only OpTC, LANL, CERT, PICO data sets are supported.')

    # Convert from str to function pointer
    if args.encoder_name != 'ARGUS':
        raise NotImplementedError("wrong encoder", args.encoder_name, args.dataset)

    if args.dataset.startswith('L'):
        if args.flows and args.fusion == 'late':
            args.encoder = detector_lanl_late_rref
        elif args.flows and args.fusion == 'uniflows':
            args.encoder = detector_lanl_uniflows_rref
        else:
            args.encoder = detector_lanl_rref
    elif args.dataset.startswith('O'):
        if args.flows and args.fusion == 'late':
            args.encoder = detector_optc_late_rref
        elif args.flows and args.fusion == 'early':
            args.encoder = detector_optc_earlyfusion_rref
        elif args.flows and args.fusion == 'uniflows':
            raise NotImplementedError("OpTC는 --fusion uniflows를 지원하지 않음")
        else:
            args.encoder = detector_optc_rref
    elif args.dataset.startswith('C'):
        args.encoder = detector_cert_rref
        if args.flows and args.fusion != 'late':
            print(f"CERT는 Late Fusion만 지원. --fusion {args.fusion} 무시")
    elif args.dataset.startswith('P'):
        args.encoder = detector_pico_late_rref
        if args.flows and args.fusion != 'late':
            print(f"PicoDomain은 Late Fusion만 지원. --fusion {args.fusion} 무시")

    if not args.flows:
        print("fusion: none (인증만 사용)")
    else:
        print(f"fusion: {args.fusion}")

    if args.rnn == 'GRU':
        args.rnn = GRU
    elif args.rnn == 'LSTM':
        args.rnn = LSTM
    else:
        args.rnn = EmptyModel
    return args, readable, model_str


if __name__ == '__main__':
    args, argstr, modelstr = args()
    if args.gpu:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    OUTPATH = './Exps/result/' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + f'_seed{args.seed}' + '/'
    if not os.path.exists(OUTPATH):
        os.makedirs(OUTPATH)
    if args.rnn != EmptyModel:
        worker_args = [args.hidden, args.hidden]
        rnn_args = [args.hidden, args.hidden, args.zdim]
    else:
        worker_args = [args.hidden, args.zdim]
        rnn_args = [None, None, None]
    stats = classification(args, rnn_args, worker_args, OUTPATH, device)
