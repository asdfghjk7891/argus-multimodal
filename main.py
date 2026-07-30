from argparse import ArgumentParser
import os, datetime
import pandas as pd
import torch
import loaders.load_optc as optc
import loaders.load_lanl as lanl
from models.recurrent import GRU, LSTM, EmptyModel
from models.argus import detector_lanl_rref, detector_optc_rref, detector_lanl_late_rref, detector_lanl_uniflows_rref
from classification import classification

# Reproducibility
import numpy as np
import random

def set_seed(seed):
    random.seed(seed)        # python random generator
    np.random.seed(seed)     # numpy random generator
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[시드 설정] seed = {seed}")

def args():
    ap = ArgumentParser()
    ap.add_argument('-d', '--delta', type=float, default=1)
    ap.add_argument('-e', '--encoder_name', type=str.upper,default="ARGUS")
    ap.add_argument('-r', '--rnn', choices=['GRU', 'LSTM', 'NONE'], type=str.upper, default="GRU")
    ap.add_argument('-H', '--hidden', type=int, default=32)
    ap.add_argument('-z', '--zdim', type=int, default=16)
    ap.add_argument('-l', '--load', action='store_true')
    ap.add_argument('--gpu', action='store_true')
    # The end of testing time, see load_lanl.TIMES
    ap.add_argument('-te', '--te_end', choices=['20', '100', '500', 'all', 'test'], type=str.lower, default="test")
    ap.add_argument('--fpweight', type=float, default=0.6)
    # For future new data sets
    ap.add_argument('--dataset', default='LANL', type=str.upper, choices=['OPTC', 'LANL'])
    ap.add_argument('--lr', default=0.01, type=float)
    ap.add_argument('--patience', default=3, type=int)
    ap.add_argument('--nratio', default=1, type=int)
    ap.add_argument('--epochs', default=100, type=int)
    # [수정] store_false → store_true: --flows 입력 시 flows 사용, 없으면 미사용
    ap.add_argument('--flows', action='store_true')
    ap.add_argument('--fusion', type=str, default='early',
                choices=['early', 'late', 'uniflows'],
                help='멀티모달 융합 방식 선택: early (기본값) 또는 late')
    ap.add_argument('--loss', type=str, default="default", choices=['default', 'ap', 'bce'])
    # [추가] 랜덤 시드를 명령어 옵션으로 지정 가능하도록 추가
    ap.add_argument('--seed', type=int, default=0,
                    help='랜덤 시드를 지정합니다. 기본값은 0입니다. '
                         '예: --seed 1')
    # [추가] 데이터 경로를 명령어 옵션으로 지정 가능하도록 추가
    ap.add_argument('--data_path', type=str, default=None,
                    help='데이터셋 경로를 직접 지정합니다. '
                         '예: --data_path C:/Users/user/Desktop/Argus/data/ '
                         '지정하지 않으면 load_optc.py 또는 load_lanl.py의 기본 경로를 사용합니다.')
    args = ap.parse_args()
    assert args.fpweight >= 0 and args.fpweight <=1, '--fpweight must be a value between 0 and 1 (inclusive)'

    # [추가] 시드 설정 적용
    set_seed(args.seed)

    readable = str(args)
    print(readable)
    model_str = '%s -> %s ' % (args.encoder_name , args.rnn)
    print(model_str)
    args.dataset = args.dataset+'_'+args.encoder_name

    # [추가] data_path 옵션이 지정된 경우 경로 끝에 슬래시(/) 자동 추가
    if args.data_path is not None:
        args.data_path = args.data_path.replace('\\', '/')
        if not args.data_path.endswith('/'):
            args.data_path += '/'

    # Parse dataset info
    if args.dataset.startswith('O'):
        # [추가] data_path가 지정된 경우 load_optc의 폴더 경로를 동적으로 변경
        if args.data_path is not None:
            optc.OPTC_FOLDER = args.data_path
        args.loader = optc.load_optc_dist
        args.tr_start = 0
        args.tr_end = optc.DATE_OF_EVIL_LANL
        args.val_times = None # Computed later
        #make the test end as an input param
        args.te_times = [(args.tr_end, optc.TIMES[args.te_end])]
        args.delta = int(args.delta * (60**2))
    elif args.dataset.startswith('L'):
        # [추가] data_path가 지정된 경우 load_lanl의 폴더 경로를 동적으로 변경
        if args.data_path is not None:
            lanl.LANL_FOLDER = args.data_path
        args.loader = lanl.load_lanl_dist
        args.tr_start = 0
        args.tr_end = lanl.DATE_OF_EVIL_LANL
        args.val_times = None # Computed later
        #make the test end as an input param
        args.te_times = [(args.tr_end, lanl.TIMES[args.te_end])]
        # args.delta = 1
        args.delta = int(args.delta * (60**2))
    else:
        raise NotImplementedError('Only OpTC and LANL data sets are supported.')

    # Convert from str to function pointer
    if (args.encoder_name == 'ARGUS') and (args.dataset.startswith('L')):
        if args.flows and args.fusion == 'late':
            args.encoder = detector_lanl_late_rref
            print("[융합 방식] Late Fusion 사용")
        elif args.flows and args.fusion == 'uniflows':
            args.encoder = detector_lanl_uniflows_rref
            print("[융합 방식] Uni-Flows 사용 (flows만 단독)")
        else:
            args.encoder = detector_lanl_rref
            print("[융합 방식] Early Fusion 사용")
    elif (args.encoder_name == 'ARGUS') and (args.dataset.startswith('O')):
        args.encoder = detector_optc_rref
    else:
        raise NotImplementedError("wrong encoder", args.encoder_name, args.dataset)

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
    OUTPATH = './Exps/result/'+ datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + f'_seed{args.seed}' + '/'
    if not os.path.exists(OUTPATH): os.makedirs(OUTPATH)
    if args.rnn != EmptyModel:
        worker_args = [args.hidden, args.hidden]
        rnn_args = [args.hidden, args.hidden, args.zdim]
    else:
        worker_args = [args.hidden, args.zdim]
        rnn_args = [None, None, None]
    stats = classification(args, rnn_args, worker_args, OUTPATH, device)
